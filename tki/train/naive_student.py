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

class CLrStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(CLrStudent, self).__init__(student_args=student_args, 
                                           supervisor=supervisor, 
                                           id=id)
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False, action=0.0):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            if not first_batch:
                if self.c_flag:
                    self.optimizer.lr = self.optimizer.lr + action
                    self.c_flag=False
                
                # lr-value protect
                if self.optimizer.lr <= 0.001:
                    self.optimizer.lr = 0.001
                elif self.optimizer.lr >= 4.0:
                    self.optimizer.lr = 4.0
                    
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics
    
class ReStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(ReStudent, self).__init__(student_args=student_args, 
                                           supervisor=supervisor, 
                                           id=id)
        self.penalty_factor = 0.0001
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False, action=1.0):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            if not first_batch:
                # self.penalty_factor += action
                grads = [g+2*self.penalty_factor*action*w for g,w in zip(gradients, self.model.trainable_variables)]
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics
    
class LrReStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(LrReStudent, self).__init__(student_args=student_args, 
                                           supervisor=supervisor, 
                                           id=id)
        self.penalty_factor = 0.0001
    
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False, action=1.0):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            if not first_batch:
                gradients = [action[0]*g for g in gradients]
                grads = [g+2*self.penalty_factor*action[1]*w for g,w in zip(gradients, self.model.trainable_variables)]
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics

class OclrStudent(Student):
    def __init__(self, student_args, supervisor = None, id = 0):
        super(OclrStudent, self).__init__(student_args=student_args, 
                                        supervisor=supervisor, 
                                        id=id)
        
    # @tf.function(experimental_relax_shapes=True, experimental_compile=None)
    def _train_step(self, inputs, labels, first_batch=False, action=0.0):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predictions))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            if not first_batch:
                if self.c_flag:
                    self.optimizer.lr = self.optimizer.lr + action
                    self.c_flag=False
                
                # lr-value protect
                if self.optimizer.lr <= 0.001:
                    self.optimizer.lr = 0.001
                elif self.optimizer.lr >= 4.0:
                    self.optimizer.lr = 4.0
                    
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.mt_loss_fn.update_state(loss)
        
        return loss, gradients, train_metrics
    
        
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
                if (epoch*train_steps_per_epoch+train_step) % valid_args['valid_gap'] == 0:
                    
                    expect_q_values, state = self.supervisor(self.model.trainable_variables)
                    act_idx = self.policy(expect_q_values, self.id)
                    action = self.act_space(act_idx)
                    self.action = action
                    self.c_flag = True
                    
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
                    # online update
                    # form a batch from replay buffer
                    self.supervisor.update(inputs=(state, act_idx), labels=q_value)
                    
                    with self.logger.as_default():
                        tf.summary.scalar("q_value", q_value, step=epoch*train_steps_per_epoch+train_step)
                        tf.summary.scalar("lr", self.optimizer.lr, step=epoch*train_steps_per_epoch+train_step)
                        if self.act_space.act_style == "2_dims":
                            tf.summary.scalar("act_lr",  action[0], step=epoch*train_steps_per_epoch+train_step)
                            tf.summary.scalar("act_re",  action[1], step=epoch*train_steps_per_epoch+train_step)
                        else:
                            tf.summary.scalar("action",  action, step=epoch*train_steps_per_epoch+train_step)
                            
            etr_loss = self.mt_loss_fn.result()
            etr_metric = self.train_metrics.result()
                    
            return etr_loss, etr_metric