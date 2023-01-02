import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import horovod.tensorflow as hvd

# others
from tqdm import trange
from easydict import EasyDict as edict

from tki.train.modules.RL.action import ActionSpace
from tki.train.modules.RL.policy import PolicySpace

from tki.train.student import Student
from tki.tools.utils import print_warning, print_green, print_error, print_normal, check_mkdir
from tki.train.utils import ForkedPdb

class HVDStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(HVDStudent, self).__init__(student_args=student_args, 
                                        supervisor=supervisor, 
                                        id=id)
        
    
    def _build_enviroment(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    def _build_optimizer(self):
        optimizer_args = self.args.optimizer
        optimizer = tf.keras.optimizers.get(optimizer_args.name)
        optimizer.learning_rate = optimizer_args.learning_rate * hvd.size()
        optimizer = hvd.DistributedOptimizer(optimizer)
        self.base_lr = optimizer_args.learning_rate

        return optimizer
    
    def _create_logdir(self):
        logdir = "tensorboard/" + "{}-{}".format(self.name, self.id) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(self.args.log_path, logdir)
        if hvd.rank() == 0:
            check_mkdir(logdir)
        return logdir
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False, action=1.0):
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
        
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        grads = [action*g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.mt_loss_fn.update_state(loss)
        
        if first_batch:
            grads = None
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
        
        self.mt_loss_fn.update_state(loss)
        
        return loss, grads, train_metrics
    
    def train_block(self, epoch, train_steps_per_epoch, train_iter, valid_args, valid_iter):
        with trange(train_steps_per_epoch, desc="Train steps", leave=False) as t:
            self.mt_loss_fn.reset_states()
            self.train_metrics.reset_states()
            for train_step in t:
                
                # train
                data = train_iter.get_next()
                first_batch = True if epoch*train_steps_per_epoch+train_step == 0 else False
                train_loss, train_gard, train_metrics = self._train_step(data['inputs'], data['labels'], 
                                                                         first_batch=first_batch, action=self.action)
                
                
                # valid
                if hvd.rank() == 0:
                    if (epoch*train_steps_per_epoch+train_step) % valid_args['valid_gap'] == 0:
                        
                        expect_q_values, state = self.supervisor(self.model.trainable_variables)
                        act_idx = self.policy(expect_q_values, self.id)
                        action = self.act_space(act_idx)
                        self.action = action
                        
                        valid_loss, valid_metrics = self.valid_block(train_step, valid_args, valid_iter)
                        
                        # training knowledge
                        q_value = self.training_knowledge.update_buffer(state=state,
                                                            expect_q_values=expect_q_values, 
                                                            train_loss=train_loss,
                                                            train_metric=train_metrics,
                                                            act_grad = train_gard,
                                                            action=action,
                                                            act_idx=act_idx,
                                                            valid_loss=valid_loss, 
                                                            valid_metric=valid_metrics,
                                                            step=epoch*train_steps_per_epoch+train_step)
                        with self.logger.as_default():
                            tf.summary.scalar("q_value", q_value, step=epoch*train_steps_per_epoch+train_step)
                            tf.summary.scalar("action",  action, step=epoch*train_steps_per_epoch+train_step)
                            
            etr_loss = self.mt_loss_fn.result()
            etr_metric = self.train_metrics.result()
            return etr_loss, etr_metric
    
    def train(self):
        
        # parse train loop control args
        train_loop_args = self.args['train_loop']
        train_args = train_loop_args['train']
        valid_args = train_loop_args['valid']

        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        
        total_epochs = self.dataloader.info['epochs']  
        train_steps_per_epoch = self.dataloader.info['train_step']

        # train, valid, test
        # tqdm update, logger
        with trange(total_epochs, desc="Epochs") as e:
            for epoch in e:
                # hparams setting
                self.hparam_tuning(train_args, epoch, total_epochs)

                # train
                etr_loss, etr_metric= self.train_block(epoch, train_steps_per_epoch, train_iter, valid_args, valid_iter)
                
                # test
                if hvd.rank() == 0:
                    ete_loss, ete_metric = self.test_block(epoch, test_iter)
                
                    e.set_postfix(etr_loss=etr_loss.numpy(), etr_metric=etr_metric.numpy(), ete_loss=ete_loss.numpy(), 
                                ete_metric=ete_metric.numpy(), lr = self.optimizer.learning_rate.numpy())
                    
                    with self.logger.as_default():
                        tf.summary.scalar("etr_loss", etr_loss, step=epoch)
                        tf.summary.scalar("etr_metric", etr_metric, step=epoch)
                        tf.summary.scalar("ete_loss", ete_loss, step=epoch)
                        tf.summary.scalar("ete_metric", ete_metric, step=epoch)
                        
        if hvd.rank() == 0:
            self.model.summary()
            self.model_save(name="finished")
        