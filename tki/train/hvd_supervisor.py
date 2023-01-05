
import os
from tqdm import trange
import tensorflow as tf
from datetime import datetime
import horovod.tensorflow as hvd

from tki.tools.utils import check_mkdir, print_green
from tki.train.utils import ForkedPdb
from tki.train.modules.RL.action import weights_augmentation
from tki.train.supervisor import Supervisor

class HVDSupervisor(Supervisor):
    def __init__(self, supervisor_args, id = 0):
        super(HVDSupervisor, self).__init__(supervisor_args, id = id)
        self._build_enviroment()
        
    def _build_enviroment(self):
        hvd.init()
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
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False):
        import pdb
        ForkedPdb().set_trace()
        with tf.GradientTape() as tape:
            states, act_idx = inputs
            predictions = self.model(states)
            predict_value = tf.gather_nd(params=predictions, indices = tf.reshape(act_idx,(-1,1)),batch_dims=1)
            loss = self.loss_fn(labels, predict_value)
            print(loss)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predict_value))
            
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        if first_batch:
            grads = None
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics
    
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
                if hvd.rank() == 0:
                    with self.logger.as_default():
                        tf.summary.scalar("etr_loss", etr_loss, step=self.id*total_epochs + epoch)
                        tf.summary.scalar("etr_metric", etr_metric, step=self.id*total_epochs + epoch)

        if hvd.rank() == 0:
            self.model.summary()
            self.model_save(name="finished")
    
    def run(self, keep_train=False, new_students=[]):
    
        if keep_train:
            # prepare dataset
            print_green("-"*10+"run_keep"+"-"*10)
            self.new_students = new_students
            self.train_dataset, self.valid_dataset, self.test_dataset \
                = self.dataloader.load_dataset(new_students = new_students)
            
            # train
            self.train()
            
        else:
            # set enviroment
            print_green("-"*10+"run_init"+"-"*10)

            # prepare dataset
            self.train_dataset, self.valid_dataset, self.test_dataset, \
            self.dataloader = self._build_dataset()

            # build optimizer
            self.optimizer = self._build_optimizer()

            # build losses and metrics
            self.loss_fn, self.mt_loss_fn, self.mv_loss_fn, self.mtt_loss_fn = self._build_loss_fn()
            self.train_metrics, self.valid_metrics, self.test_metrics = self._build_metrics()
            
            # build weights and writter
            self.logger = self._build_logger()

            # train
            self.train()
            
        self.id += 1
