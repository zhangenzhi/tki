import tensorflow as tf
from rebyval.tools.utils import *
from rebyval.train.base_trainer import BaseTrainer


class TargetTrainer(BaseTrainer):
    def __init__(self, trainer_args,surrogate_model):
        super(TargetTrainer, self).__init__(trainer_args=trainer_args)
        self.surrogate_model = surrogate_model

    def reset_dataset(self):
        if self.args['dataloader']['name'] == 'cifar10':
            train_dataset, valid_dataset, test_dataset = self.dataloader.load_dataset()
            return train_dataset, valid_dataset, test_dataset

    @BaseTrainer.timer
    def during_train(self):
        try:
            x = self.train_iter.get_next()
            y = x.pop('label')
            if self.train_args['rebyval']:
                self._train_step_rebyval(x,y)
            else:
                self._train_step(x, y)
        except:
            print_warning("during traning exception")
            self.epoch += 1
            self.train_dataset, _, _ = self.reset_dataset()
            self.train_iter = iter(self.train_dataset)

    @BaseTrainer.timer
    def during_valid(self):
        try:
            x_valid = self.valid_iter.get_next()
            y_valid = x_valid.pop('label')
            self._valid_step(x_valid, y_valid)

        except:
            print_warning("during validation exception")
            _,self.valid_dataset,_ = self.reset_dataset()
            self.valid_iter = iter(self.valid_dataset)

    @BaseTrainer.timer
    def during_test(self):
        print('[Testing Status]: Test Step: {:04d}'.format(self.test_step))

        try:
            x_test = self.test_iter.get_next()
            y_test = x_test.pop('label')
            self._test_step(x_test, y_test)
        except:
            self.test_flag = False

    @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step_rebyval(self, inputs, labels):
        try:
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                t_loss = self.metrics['loss_fn'](labels, predictions)
                v_loss = self.surrogate_model({'inputs':self.model.trainable_variables})
                loss = t_loss + v_loss

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            self.metrics['train_loss'](loss)
        except:
            print_error("rebyval train step error")
            raise

    def run_with_weights_collect(self):
        if self.args['train_loop_control']['valid']['analyse']:
            self.run()
        else:
            print_error("analysis not open in Target Trainer")
            raise("error")

class SurrogateTrainer(BaseTrainer):
    def __init__(self, trainer_args):
        super(SurrogateTrainer, self).__init__(trainer_args=trainer_args)

    def reset_dataset(self):
        if self.args['dataloader']['name'] == 'dnn_weights':
            train_dataset, valid_dataset, test_dataset = self.dataloader.load_dataset()
            return train_dataset, valid_dataset, test_dataset

    @BaseTrainer.timer
    def during_train(self):
        try:

            x = self.train_iter.get_next()
            y = x.pop('valid_loss')
            x.pop('train_loss')
            x.pop('vars_length')
            flat_vars = []
            for feat,tensor in x.items():
                flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0],-1)))
            flat_vars = tf.concat(flat_vars, axis=1)
            flat_input = {'inputs':flat_vars}
            self._train_step(flat_input, y)
        except:
            print_warning("during traning exception")
            self.epoch += 1
            self.train_dataset, _, _ = self.reset_dataset()
            self.train_iter = iter(self.train_dataset)

    @BaseTrainer.timer
    def during_valid(self):
        try:
            x_valid = self.valid_iter.get_next()
            y_valid = x_valid.pop('valid_loss')
            x_valid.pop('train_loss')
            x_valid.pop('vars_length')
            flat_vars = []
            for feat,tensor in x_valid.items():
                flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0],-1)))
            flat_vars = tf.concat(flat_vars, axis=1)
            flat_input = {'inputs':flat_vars}
            self._valid_step(flat_input, y_valid)

        except:
            print_warning("during validation exception")
            _, self.valid_dataset, _ = self.reset_dataset()
            self.valid_iter = iter(self.valid_dataset)

    @BaseTrainer.timer
    def during_test(self):

        print('[Testing Status]: Test Step: {:04d}'.format(self.test_step))

        try:
            x_test = self.test_iter.get_next()
            y_test = x_test.pop('valid_loss')
            x_test.pop('train_loss')
            x_test.pop('vars_length')
            flat_vars = []
            for feat, tensor in x_test.items():
                flat_vars.append(tf.reshape(tensor, shape=(tensor.shape[0], -1)))
            flat_vars = tf.concat(flat_vars, axis=1)
            flat_input = {'inputs': flat_vars}

            self._test_step(flat_input, y_test)
        except:
            self.test_flag = False
