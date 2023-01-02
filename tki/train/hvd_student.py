import numpy as np
import tensorflow as tf

# others
from tqdm import trange
from easydict import EasyDict as edict

from tki.train.modules.RL.action import ActionSpace
from tki.train.modules.RL.policy import PolicySpace

from tki.train.student import Student
from tki.tools.utils import print_warning, print_green, print_error, print_normal

class HVDStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(HVDStudent, self).__init__(student_args=student_args, 
                                           supervisor=supervisor, 
                                           id=id)
        
    
    def _tki_train_step(self, inputs, labels, action):
        # base direction 
        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            train_loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            
        train_gard = tape_t.gradient(train_loss, self.model.trainable_variables)
        self.mt_loss_fn.update_state(train_loss)

        # next state
        gradients = [g*action for g in train_gard]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
        return train_loss, train_gard, train_metrics
        