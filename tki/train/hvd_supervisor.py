
import os
from tqdm import trange
import tensorflow as tf
from datetime import datetime
import horovod.tensorflow as hvd
hvd.init()

from tki.tools.utils import check_mkdir
from tki.train.modules.RL.action import weights_augmentation
from tki.train.supervisor import Supervisor

class HVDSupervisor(Supervisor):
    def __init__(self, supervisor_args, id = 0):
        super(HVDSupervisor, self).__init__(supervisor_args, id = id)
        
    def _build_enviroment(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    def update(self, inputs, labels):
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            predictions = tf.squeeze(predictions)
            labels = tf.reshape(labels,predictions.shape)
            loss = self.loss_fn(labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        print(loss)
    
    def _create_logdir(self):
        logdir = "tensorboard/" + "{}-{}".format(self.name, self.id) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(self.args.log_path, logdir)
        if hvd.rank() == 0:
            check_mkdir(logdir)
        return logdir
    
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
                etr_loss, etr_metric= self.train_block(epoch, train_steps_per_epoch, train_iter)
                    
                e.set_postfix(etr_loss=etr_loss.numpy(), etr_metric=etr_metric.numpy())
                
                with self.logger.as_default():
                    tf.summary.scalar("etr_loss", etr_loss, step=self.id*total_epochs + epoch)
                    tf.summary.scalar("etr_metric", etr_metric, step=self.id*total_epochs + epoch)

        if hvd.rank() == 0:
            self.model.summary()
            self.model_save(name="finished")
