import os
from tqdm import trange
import tensorflow as tf

# others
from tki.train.modules.RL.action import weights_augmentation
from tki.tools.utils import print_green, print_error, print_normal, check_mkdir
from tki.train.trainer import Trainer

class Supervisor(Trainer):
    def __init__(self, supervisor_args, id = 0):
        super(Supervisor, self).__init__(trainer_args=supervisor_args, id=id)
        
        # build model
        self._build_enviroment()
        self.model = self._build_model()
        self.name = "sp_" + supervisor_args.name
        
        self.args.dataloader.path = os.path.join(supervisor_args.log_path, "weight_space")
        self.weights_style = supervisor_args.train_loop.train.weights_style

    def weights_augmentation(self, weights):
        return weights_augmentation(weights=weights, style=self.weights_style) 
    
    def __call__(self, weights):
        state = self.weights_augmentation(weights=weights)
        prediction = self.model(tf.expand_dims(state, axis=0), training=False)
        return prediction, state
    
    def update(self, inputs, labels):
        
        with tf.GradientTape() as tape:
            states, act_idx = inputs
            predictions = self.model(states)
            predict_value = tf.gather_nd(params=predictions, indices = tf.reshape(act_idx,(-1,1)),batch_dims=1)
            loss = self.loss_fn(labels, predict_value)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predict_value))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, gradients, train_metrics
        
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False):
        
        with tf.GradientTape() as tape:
            states, act_idx = inputs
            predictions = self.model(states)
            predict_value = tf.gather_nd(params=predictions, indices = tf.reshape(act_idx,(-1,1)),batch_dims=1)
            loss = self.loss_fn(labels, predict_value)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predict_value))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics
    
    def train_block(self, epoch, train_steps_per_epoch, train_iter):
        with trange(train_steps_per_epoch, desc="Train steps", leave=False) as t:
            self.mt_loss_fn.reset_states()
            self.train_metrics.reset_states()
            for train_step in t:
                # train
                data = train_iter.get_next()
                train_loss, gard, train_metrics = self._train_step((data['state'], data['act_idx']), data['reward'])
                            
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
                etr_loss, etr_metric= self.train_block(epoch, train_steps_per_epoch, train_iter)
                # test
                # ete_loss, ete_metric = self.test_block(epoch, test_iter)
                    
                e.set_postfix(etr_loss=etr_loss.numpy(), etr_metric=etr_metric.numpy())
                
                with self.logger.as_default():
                    tf.summary.scalar("etr_loss", etr_loss, step=self.id*total_epochs + epoch)
                    tf.summary.scalar("etr_metric", etr_metric, step=self.id*total_epochs + epoch)
                    # tf.summary.scalar("ete_loss", ete_loss, step=epoch)
                    # tf.summary.scalar("ete_metric", ete_metric, step=epoch)
        
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

