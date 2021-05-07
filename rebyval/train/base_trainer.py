import os
import time
import tensorflow as tf
from functools import wraps

# dataloader
from rebyval.dataloader.dataset_loader import Cifar10DataLoader
from rebyval.dataloader.weights_loader import DnnWeightsLoader

# model
from rebyval.model.dnn import DenseNeuralNetwork

# optimizer
from rebyval.optimizer.scheduler.linear_scaling_with_warmup import LinearScalingWithWarmupSchedule

# others
from rebyval.train.utils import get_scheduler,prepare_dirs
from rebyval.tools.utils import calculate_auc, write_log, print_green, print_error, print_normal


# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.random.set_random_seed(1024)
# tf.debugging.set_log_device_placement(enabled=True)


class BaseTrainer:
    def __init__(self, trainer_args):
        self.args = trainer_args

        # create timer collection
        member_func = [
            func for func in dir(self)
            if callable(getattr(self, func)) and not func.startswith("__")
        ]
        expanded_func = ["cumulate_" + func for func in member_func]
        self.timer_dict = dict(zip(expanded_func, [0] * len(expanded_func)))

        # create during value collection
        self.during_value_dict = {}

    @classmethod
    def timer(cls, func):
        @wraps(func)
        def wrapper(self, *arg, **kwds):
            start_time = time.time()
            res = func(self, *arg, **kwds)
            end_time = time.time()
            cost_time = end_time - start_time
            self.timer_dict[func.__name__] = cost_time
            self.timer_dict["cumulate_" + func.__name__] += cost_time
            return res

        return wrapper

    def _build_dataset(self):

        dataset_args = self.args['dataloader']
        train_dir = test_dir = valid_dir = ""

        if dataset_args['name'] == 'cifar10':
            dataloader = Cifar10DataLoader(dataset_args)
            train_dataset, valid_dataset, test_dataset = dataloader.load_dataset()
        elif dataset_args['name'] == 'dnn_weights':
            dataloader = DnnWeightsLoader(dataset_args)
            train_dataset, valid_dataset, test_dataset = dataloader.load_dataset()
        else:
            print_error("no such dataset:{}".format(dataset_args['name']))
            raise("no such dataset")

        self.dataset_path = {
            "train_dir": train_dir,
            "valid_dir": valid_dir,
            "test_dir": test_dir
        }
        return train_dataset, valid_dataset, test_dataset, dataloader

    def _build_model(self):
        model_args = self.args['model']
        if model_args['name'] == 'dnn':
            deep_dims = list(map(lambda x: float(x), model_args['deep_dims'].split(',')))
            model = DenseNeuralNetwork(deep_dims=deep_dims)
        else:
            print_error("no such model: {}".format(model_args['name']))
            raise("no such model")

        return model

    def _build_losses(self):
        metrics = dict()

        metrics['loss_fn'] = tf.keras.losses.get(self.args['loss']['name'])

        metrics['train_loss'] = tf.keras.metrics.Mean(name='train_loss')
        metrics['train_accuracy'] = tf.keras.metrics.AUC(name='train_auc')
        metrics['valid_loss'] = tf.keras.metrics.Mean(name='valid_loss')
        metrics['valid_accuracy'] = tf.keras.metrics.AUC(name='valid_auc')
        metrics['test_loss'] = tf.keras.metrics.Mean(name='test_loss')
        metrics['test_accuracy'] = tf.keras.metrics.AUC(name='test_auc')

        return metrics

    def _build_enviroment(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def _build_optimizer(self):
        optimizer_args = self.args['optimizer']
        learning_rate = float(optimizer_args['learning_rate'])

        if optimizer_args['name'] == 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=learning_rate)
        elif optimizer_args['name'] == 'SGD':
            if optimizer_args['scheduler'] == 'linear_scaling_with_warmup':
                linear_scaling = self.args.examples_per_parsing if self.args.examples_per_parsing else 1
                scheduler = LinearScalingWithWarmupSchedule(linear_scaling=linear_scaling,
                                                            base_learning_rate=learning_rate,
                                                            warmup_steps=3000,
                                                            gradual_steps=80000)
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=scheduler)
            else:
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=learning_rate)
        elif optimizer_args['name'] == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=learning_rate)

        elif optimizer_args['name'] == 'Adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate)

        elif optimizer_args['name'] == 'Ftrl':
            optimizer = tf.keras.optimizers.Ftrl(
                learning_rate=learning_rate)

        elif optimizer_args['name'] == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate=learning_rate)

        elif optimizer_args['name'] == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate)
        else:
            raise print_error(
                '[Optimizer Status]: Unspport optimizer type: {:}'.format(
                    optimizer_args['name']))

        optimizer_msg = '[Optimizer]: {:} with lr={:}'.format(optimizer_args['name'],
                                                              optimizer_args['learning_rate'])
        print_green(optimizer_msg)

        return optimizer

    def _build_optimier_from_keras_get(self):
        optimizer_args = self.args['optimizer']
        optimzier = tf.keras.optimizers.get(optimizer_args['name'])
        if optimizer_args['scheduler']:
            scheduler_args = optimizer_args['scheduler']
            scheduler = get_scheduler(scheduler_args)
            optimzier.learning_rate = scheduler
        else:
            optimzier.learning_rate = optimizer_args['learning_rate']

        optimizer_msg = '[Optimizer]: {:} with lr={:}'.format(optimizer_args['name'],
                                                              optimizer_args['learning_rate'])
        print_green(optimizer_msg)

        return optimzier

    def reset_dataset(self):
        dataset = self.dataloader.load_dataset()
        return dataset

    def model_restore(self):
        step = 0
        best_auc = 0
        model_args = self.args['model']
        model_args['restore_model'] = None
        if model_args['restore_model']:
            model_latest = os.path.join(model_args['restore_model']['restore_model_dir'],
                                        'model_latest')
            if os.path.exists(model_latest + '.data-00000-of-00001') \
                    and os.path.exists(model_latest + '.index'):

                with open(self.args.log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        cur_auc = float(line.split(':')[-1][:-1])
                        if cur_auc > best_auc:
                            best_auc = cur_auc
                    step = int(lines[-1].split(':')[1])

                self.model.load_weights(model_latest)
                print(
                    '\033[33m[Model Status]: Restore training step:{:08d} from {:}.\033[0m'
                        .format(step, model_latest))
                return step, best_auc

            else:
                print(
                    '\033[33m[Model Status]: Restoring Model failed, {:} not exist.\033[0m'
                        .format(model_latest))
        else:
            print('\033[33m[Model Status]: Training from Scratch ...\033[0m')
        return step, best_auc

    def model_save_by_name(self, name):
        save_path = os.path.join(self.valid_args['model_dir'], self.valid_args['save_model']['save_in'])
        save_path = os.path.join(save_path, 'model_{}'.format(name))

        save_msg = '\033[33m[Model Status]: Saving {} model at step:{:08d} in {:}.\033[0m'.format(
            name, self.global_step, save_path)
        print(save_msg)
        self.model.save_weights(save_path,
                                overwrite=True,
                                save_format='tf')

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels):
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.metrics['loss_fn'](labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            self.metrics['train_loss'](loss)
        except:
            print_error("train step error")
            raise

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _valid_step(self, inputs, labels):
        try:
            predictions = self.model(inputs, training=False)
            v_loss = self.metrics['loss_fn'](labels, predictions)

            self.metrics['valid_loss'](v_loss)
            return predictions
        except:
            print_error("valid step erroe")
            raise


    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _test_step(self, inputs, labels):
        try:
            predictions = self.model(inputs, training=False)
            t_loss = self.metrics['loss_fn'](labels, predictions)

            self.metrics['test_loss'](t_loss)

            return predictions
        except:
            print_error("test step error")
            raise

    def run(self, set_args=None):

        # set enviroment
        self._build_enviroment()

        # prepare dataset
        self.train_dataset, self.valid_dataset, self.test_dataset, \
        self.dataloader = self._build_dataset()

        # build optimizer
        self.optimizer = self._build_optimizer()

        # build model
        self.model = self._build_model()

        # build losses and metrics
        self.metrics = self._build_losses()

        # train
        train_msg = self.main_loop()

        return train_msg

    # Experiment config
    def before_exp(self):

        # We design the preparing work from two levels:
        #
        # 1. Initialize the main components used in train/test/validation.
        #    for example, model, dataloader & dataset, loss functioin,
        #    optimizer, and enviroment etc. The functions for "build" these
        #    components mainly start with "_build_*"
        #
        # 2. Verifying or reset the infomation for each component. for example
        #    reset loss, check & restore model, used information in training step,
        #    make dataset iterable and etc.
        #
        # This function is the last validation before main training procedure.
        # which means all the elements should be ready here.

        # parse train loop control args
        train_loop_control_args = self.args['train_loop_control']
        self.train_args = train_loop_control_args['train']
        self.valid_args = train_loop_control_args['valid']
        self.test_args = train_loop_control_args['test']

        # model restore
        self.global_step = 0
        self.global_step, self.global_best_auc = self.model_restore()

        # dataset train, valid
        self.epoch = 0
        self.train_iter = iter(self.train_dataset)
        self.valid_iter = iter(self.valid_dataset)
        self.test_iter = iter(self.test_dataset)

        # numerical reset
        self.metrics['train_loss'].reset_states()
        self.metrics['train_accuracy'].reset_states()

        # log collection flags
        self.init_step = self.global_step

        # prepare_dirs
        prepare_dirs(valid_args=self.valid_args)

    # Train
    def before_train(self):

        try:
            # numerical reset
            self.auc_list = []
            self.step_list = []

            # model restore
            self.train_step = 0

        except:
            raise ValueError

    def during_train(self):
        raise NotImplementedError(
            'Trainer must implement build_model function.')

    def after_train_step(self):
        # incress train && global steps
        self.train_step += 1

        # message print

        iter_msg = '[Training Status]: step={:08d}, train loss={:.4f}' \
                       .format(self.train_step, self.metrics['train_loss'].result()
                               ) + ', traning_current_time={:.4f}s, training_cumulative_time={:.4f}h' \
                       .format(self.timer_dict['during_train'], self.timer_dict['cumulate_during_train'] / 3600)
        print(iter_msg)

    def after_train(self):
        pass

    def train_stop_condition(self) -> bool:
        if self.train_step == 0:
            return False
        else:
            return self.train_step % self.valid_args['valid_gap'] == 0

    # Valid
    def before_valid(self):

        try:
            # prepare stop indicator
            self.valid_step = 0
            self.valid_flag = True

            # numerical reset
            self.valid_metrics_list = []
            self.metrics['valid_loss'].reset_states()
            self.metrics['valid_accuracy'].reset_states()

        except:
            raise ValueError

    def during_valid(self):
        raise NotImplementedError(
            'Trainer must implement during_valid function.')

    def after_valid_step(self):

        # increase step
        self.valid_step += 1

        # valid log collection
        valid_msg = '[Validating Status]: Valid Step: {:04d}'.format(self.valid_step)
        print(valid_msg)

        valid_auc_numpy = self.metrics['valid_accuracy'].result().numpy()
        self.valid_metrics_list.append(valid_auc_numpy)

    def after_valid(self):

        # record valid metric
        valid_auc = sum(self.valid_metrics_list) / len(self.valid_metrics_list)
        self.auc_list.append(valid_auc)

        # valid log collection
        valid_msg = 'ValidInStep :{:08d}: Epoch:{:03d}: Loss :{:.6f}: AUC :{:.6f}: ' \
            .format((self.global_step + 1) * self.valid_args['valid_gap'], self.epoch, self.metrics['valid_loss'].result(),
                    valid_auc)
        print(valid_msg)
        time_msg = 'Timer: CumulativeTraining :{:.4f}h: AvgBatchTraining :{:.4f}s: TotalCost :{:.4f}h' \
            .format(self.timer_dict['cumulate_during_train'] / 3600,
                    self.timer_dict['cumulate_during_train'] / (
                            (self.global_step + 1) * self.valid_args['valid_gap'] - self.init_step),
                    (self.timer_dict['cumulate_during_train'] + self.timer_dict['cumulate_during_valid']) / 3600)
        print(time_msg)

        # save model best
        if self.valid_args['save_model']:
            if valid_auc >= max(self.auc_list):
                self.model_save_by_name(name="best")

        # collect analyse data
        if self.valid_args['analyse'] == True:
            self.during_value_dict['vars'] = self.model.trainable_variables
            self.during_value_dict['train_loss'] = self.metrics['train_loss'].result()
            self.during_value_dict['valid_loss'] = self.metrics['valid_loss'].result()
            self._write_analyse_to_tfrecord()

        write_log(self.valid_args['log_file'], valid_msg)
        write_log(self.valid_args['log_file'], time_msg)

    def check_should_valid(self) -> bool:
        return self.test_args['check_should_test']

    def valid_stop_condition(self):
        return self.valid_step >= self.valid_args['valid_steps']

    # Test
    def before_test(self):

        try:
            # prepare stop indicator
            self.test_step = 0
            self.test_flag = True

            # numerical reset
            self.metrics['test_loss'].reset_states()
            self.metrics['test_accuracy'].reset_states()
        except:
            raise ValueError

    def during_test(self):
        raise NotImplementedError(
            'Trainer must implement build_model function.')

    def check_should_test(self) -> bool:
        return True

    def after_test_step(self):
        # increase step
        self.test_step += 1

    def after_test(self):

        # test log collection
        test_msg = 'TestInStep :{:08d}: Loss :{:.6f}: AUC :{:.6f}' \
            .format(self.global_step, self.metrics['test_loss'].result(), self.metrics['test_accuracy'].result())
        print(test_msg)
        time_msg = 'Timer: CumulativeTraining :{:.4f}h: AvgBatchTraining :{:.4f}s: TotalCost :{:.4f}h' \
            .format(self.timer_dict['cumulate_during_train'] / 3600,
                    self.timer_dict['cumulate_during_train'] / (self.global_step - self.init_step),
                    (self.timer_dict['cumulate_during_train'] + self.timer_dict['cumulate_during_valid'] +
                     self.timer_dict['cumulate_during_test']) / 3600)
        print(time_msg)

        test_auc_numpy = self.metrics['test_accuracy'].result().numpy()
        self.auc_list.append(test_auc_numpy)
        self.step_list.append(self.global_step)

        write_log(self.valid_args['log_file'], test_msg)
        write_log(self.valid_args['log_file'], time_msg)
        print(test_msg)
        print(time_msg)

    def test_stop_condition(self) -> bool:
        return self.test_flag == False

    def after_exp_step(self):
        # increase global_step
        self.global_step += 1

    def exp_stop_condition(self) -> bool:
        return self.global_step >= int(self.train_args['max_training_steps'] / self.valid_args['valid_gap'])

    def after_exp(self):
        print('[Training Status]: Training Done at Step: {:}'.format(
            self.global_step))

        tf.keras.backend.clear_session()
        return_dict = {
            'auc_list': self.auc_list,
            'step_list': self.step_list,
            'test_auc': self.metrics['test_accuracy'].result().numpy()
        }

        return return_dict

    def main_loop(self):

        # Experiment
        self.before_exp()
        while not self.exp_stop_condition():

            # Train
            self.before_train()
            while not self.train_stop_condition():
                self.during_train()
                self.after_train_step()
            self.after_train()

            # Valid
            if self.check_should_valid():
                self.before_valid()
                while not self.valid_stop_condition():
                    self.during_valid()
                    self.after_valid_step()
                self.after_valid()

            self.after_exp_step()

        # Test
        if self.check_should_test():
            self.before_test()
            while not self.test_stop_condition():
                self.during_test()
                self.after_test_step()
            self.after_test()

        return self.after_exp()

    # Analyse
    def _during_vars_example(self):
        feature = {}
        for feature_name, value in self.during_value_dict.items():
            if isinstance(value, list):
                value = [tf.io.serialize_tensor(v).numpy() for v in value]
                v_len = len(value)
                for i in range(v_len):
                    feature[feature_name + "_{}".format(i)] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value[i]]))
                feature[feature_name + "_length"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[v_len])
                )
            else:
                value = [value.numpy()]
                feature[feature_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _write_analyse_to_tfrecord(self):
        filepath = self.valid_args['analyse_dir']

        record_file = '{}.tfrecords'.format(self.global_step)
        record_file = os.path.join(filepath, record_file)

        with tf.io.TFRecordWriter(record_file) as writer:
            example = self._during_vars_example()
            writer.write(example.SerializeToString())