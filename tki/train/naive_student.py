import numpy as np
import tensorflow as tf

# others
from tqdm import trange
from easydict import EasyDict as edict

from tki.train.modules.RL.action import ActionSpace
from tki.train.modules.RL.policy import PolicySpace

from tki.train.student import Student
from tki.tools.utils import print_warning, print_green, print_error, print_normal

class NaiveStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(NaiveStudent, self).__init__(student_args=student_args, 
                                           supervisor=supervisor, 
                                           id=id)
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False, action=1.0):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            if not first_batch:
                grads = [action*g for g in gradients]
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics

class ReStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(ReStudent, self).__init__(student_args=student_args, 
                                           supervisor=supervisor, 
                                           id=id)
        self.penalty_factor = 1e-4
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False, action=1.0):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            if not first_batch:
                grads = [g+2*action*self.penalty_factor*w for g,w in zip(gradients, self.model.trainable_variables)]
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics