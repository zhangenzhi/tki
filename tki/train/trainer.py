import os
from tqdm import trange
from datetime import datetime
import tensorflow as tf


# dataloader
from tki.dataloader.factory import dataset_factory

# model
from tki.model.factory import model_factory

# others
from tki.plot.visualization import visualization
from tki.tools.utils import print_green, print_error, print_normal, check_mkdir, save_yaml_contents
from tki.dataloader.utils import glob_tfrecords


class Trainer(object):
    def __init__(self, trainer_args, id):
        self.args = trainer_args
        self.name = self.__class__.__name__
        self.id = id
        self.best_metrics = 0.0
        self.logdir = self._create_logdir()

    def _build_enviroment(self):
  
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print_green("devices:", gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def _build_dataset(self):
        dataset_args = self.args.dataloader
        dataloader = dataset_factory(dataset_args)

        train_dataset, valid_dataset, test_dataset = dataloader.load_dataset()
        return train_dataset, valid_dataset, test_dataset, dataloader
    
    def _reset_dataset(self):
        self.train_dataset, self.valid_dataset, self.test_dataset, self.dataloader = self._build_dataset()
        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        return train_iter, valid_iter, test_iter

    def _build_model(self):
        model = model_factory(self.args['model'])
  
        # model restore
        if self.args['model'].get('restore_model'):
            self.model = self.model_restore(self.model)
        return model

    def _build_loss_fn(self):
        loss_fn = {}
        loss_fn = tf.keras.losses.get(self.args.loss.name)
        mt_loss_fn = tf.keras.metrics.Mean()
        mv_loss_fn = tf.keras.metrics.Mean()
        mtt_loss_fn = tf.keras.metrics.Mean()
        return loss_fn, mt_loss_fn, mv_loss_fn, mtt_loss_fn

    def _build_metrics(self):
        # metrics = {}
        metrics = self.args.metrics
        train_metrics = tf.keras.metrics.get(metrics['name'])
        valid_metrics = tf.keras.metrics.get(metrics['name'])
        test_metrics = tf.keras.metrics.get(metrics['name'])
        return train_metrics, valid_metrics, test_metrics

    def _build_optimizer(self):
        optimizer_args = self.args.optimizer
        optimizer = tf.keras.optimizers.get(optimizer_args.name)
        optimizer.learning_rate = optimizer_args.learning_rate
        self.base_lr = optimizer_args.learning_rate

        return optimizer

    def _create_logdir(self):
        logdir = "tensorboard/" + "{}-{}".format(self.name, self.id) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(self.args.log_path, logdir)
        check_mkdir(logdir)
        return logdir
    
    def _build_logger(self):
        logger = tf.summary.create_file_writer(self.logdir)
        return logger

    def model_save(self, name):
        save_path = os.path.join(self.logdir, '{}_{}'.format(type(self.model).__name__, name))
        save_msg = '\033[33m[Model Status]: Saving {} model in {:}.\033[0m'.format(name, save_path)
        print(save_msg)
        self.model.save(save_path, overwrite=True, save_format='tf')
        
    def hparam_tuning(self, train_args, epoch, total_epochs):
        # lr decay
        if train_args["lr_decay"]:
            if epoch == int(0.5*total_epochs):
                self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                print_green("Current decayed learning rate is {}".format(self.optimizer.learning_rate.numpy()))
            elif epoch == int(0.75*total_epochs):
                self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                print_green("Current decayed learning rate is {}".format(self.optimizer.learning_rate.numpy()))
        

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False):
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)
        valid_metrics = tf.reduce_mean(self.valid_metrics(labels, predictions))
        self.mv_loss_fn.update_state(loss)
        return loss, valid_metrics

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(labels, predictions)
        test_metrics = tf.reduce_mean(self.test_metrics(labels, predictions))
        self.mtt_loss_fn.update_state(loss)
        return loss, test_metrics
        
    def valid_block(self, train_step, valid_args, valid_iter):
        
        with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
            self.mv_loss_fn.reset_states()
            for valid_step in v:
                v_data = valid_iter.get_next()
                valid_loss, valid_metrics = self._valid_step(v_data['inputs'], v_data['labels'])
                v.set_postfix(sv_loss=valid_loss.numpy())
            ev_loss = self.mv_loss_fn.result()
        return ev_loss, valid_metrics

    def train_block(self, epoch, train_steps_per_epoch, train_iter, valid_args, valid_iter):
        with trange(train_steps_per_epoch, desc="Train steps", leave=False) as t:
            self.mt_loss_fn.reset_states()
            self.train_metrics.reset_states()
            for train_step in t:
                # valid
                if train_step % valid_args.valid_gap == 0 and valid_args.valid_gap>0:
                    self.valid_block(train_step, valid_args, valid_iter)
                    
                # train
                data = train_iter.get_next()
                train_loss, gard, train_metrics = self._train_step(data['inputs'], data['labels'])
                            
            etr_loss = self.mt_loss_fn.result()
            etr_metric = self.train_metrics.result()
            return etr_loss, etr_metric
    
    def test_block(self, epoch, test_iter):
        with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
            self.mtt_loss_fn.reset_states()
            self.test_metrics.reset_states()
            for test_step in t:
                t_data = test_iter.get_next()
                t_loss, test_metrics = self._test_step(t_data['inputs'], t_data['labels'])
                t.set_postfix(test_loss=t_loss.numpy())
            ete_loss = self.mtt_loss_fn.result()
            ete_metric = self.test_metrics.result()
            
            # save best mdoel
            if self.best_metrics < ete_metric and epoch%10==0:
                self.model_save(name="best")
                self.best_metrics = ete_metric
            return ete_loss, ete_metric

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
                ete_loss, ete_metric = self.test_block(epoch, test_iter)
                    
                e.set_postfix(etr_loss=etr_loss.numpy(), etr_metric=etr_metric.numpy(), ete_loss=ete_loss.numpy(), 
                              ete_metric=ete_metric.numpy(), lr = self.optimizer.learning_rate.numpy())
                
                with self.logger.as_default():
                    tf.summary.scalar("etr_loss", etr_loss, step=epoch)
                    tf.summary.scalar("etr_metric", etr_metric, step=epoch)
                    tf.summary.scalar("ete_loss", ete_loss, step=epoch)
                    tf.summary.scalar("ete_metric", ete_metric, step=epoch)
        
        self.model.summary()
        self.model_save(name="finished")

    def run(self, connect_queue=None, devices='1'):

        # set enviroment
        self._build_enviroment(devices=devices)

        # prepare dataset
        self.train_dataset, self.valid_dataset, self.test_dataset, \
            self.dataloader = self._build_dataset()

        # build optimizer
        self.optimizer = self._build_optimizer()

        # build model
        self.model = self._build_model()

        # build losses and metrics
        self.loss_fn, self.mt_loss_fn, self.mv_loss_fn, self.mtt_loss_fn = self._build_loss_fn()
        self.train_metrics, self.valid_metrics, self.test_metrics = self._build_metrics()
        
        # build weights save writter
        self.logger = self._build_logger()

        self.train()
        print('Finished training student {}'.format(self.id))

        # if connect_queue != None:
        #     connect_queue.put(weight_dir)

        return self.weight_dir


    
        
            