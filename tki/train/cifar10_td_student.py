from tqdm import trange
import numpy as np
import tensorflow as tf

# others
import time
from tki.train.student import Student
from tki.tools.utils import print_warning, print_green, print_error, print_normal

class Cifar10TDStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(Cifar10TDStudent, self).__init__(student_args, supervisor,id)
        
        self.index_max = 0
        self.act_idx = []
        self.gloabl_train_step = 0
        self.valid_gap = 100
        self.epsilon = 0.5
        
    
    def fix_action(self, t_grad):
        flat_weights = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
        state = tf.reshape(tf.concat(flat_weights, axis=0), (1,1,-1))
        self.values = self.supervisor({'state':state})
        return self.action_sample, self.values
    
    def greedy_policy(self, values):
        if self.id%10==0:
            return 1
        else:
            return max(range(len(values)), key=values.__getitem__) 
    
    def e_greedy_policy(self, values):
        
        roll = np.random.uniform()
        if roll < self.epsilon:
            return np.random.randint(len(values))
        else:
            return max(range(len(values)), key=values.__getitem__) 
    
    def _rl_train_step(self, inputs, labels):

        # base direction 
        with tf.GradientTape() as tape_t:
            predictions = self.model(inputs, training=True)
            t_loss = self.loss_fn(labels, predictions)
        t_grad = tape_t.gradient(t_loss, self.model.trainable_variables)
                
        # fixed action with pseudo sgd
        if (self.gloabl_train_step %  self.valid_gap )==0:
            self.pre_act = 1.0 if self.gloabl_train_step<self.valid_gap else self.action_sample[self.greedy_policy(self.values)]
            self.action_sample, self.values = self.fix_action(t_grad=t_grad)
            self.index_max = self.e_greedy_policy(self.values)
            self.act_idx.append(self.index_max) 

        # next state
        act = self.action_sample[self.index_max]
        alpha = (self.gloabl_train_step %  self.valid_gap)/self.valid_gap
        smoothed_act = (1-alpha)*self.pre_act + alpha*act
        gradients = [g*smoothed_act for g in t_grad]
        act = tf.squeeze(smoothed_act)
        
        clip_grads = [tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(clip_grads, self.model.trainable_variables))
            
        self.mt_loss_fn.update_state(t_loss)
        
        reduced_grads = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in gradients], axis=-1)
        E_q = tf.squeeze(self.values[self.index_max])
        return t_loss, E_q, act, reduced_grads, self.values

    def train(self, new_student=None, supervisor_info=None):
        
        # action_sample
        self.action_sample = tf.reshape(tf.constant([0.1,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,10.0], dtype=tf.float32),shape=(-1,1))
        
        # parse train loop control args
        train_loop_args = self.args['train_loop']
        self.train_args = train_loop_args['train']
        self.valid_args = train_loop_args['valid']
        self.valid_gap = self.valid_args['valid_gap']
        self.test_args = train_loop_args['test']

        # dataset train, valid, test
        train_iter = iter(self.train_dataset)
        valid_iter = iter(self.valid_dataset)
        test_iter = iter(self.test_dataset)
        
        if supervisor_info != None:
            self.supervisor = self._build_supervisor_from_vars(supervisor_info)

        # train, valid, write to tfrecords, test
        # tqdm update, logger
        with trange(self.dataloader.info['epochs'], desc="Epochs") as e:
            for epoch in e:
                # Train
                with trange(self.dataloader.info['train_step'], desc="Train steps", leave=False) as t:
                    self.mt_loss_fn.reset_states()
                    for train_step in t:
                        data = train_iter.get_next()
                        if self.supervisor == None:
                            train_loss, grads = self._train_step(data['inputs'], data['labels'])
                            act_grad = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in grads], axis=-1)
                            action = tf.ones(shape=act_grad.shape, dtype=tf.float32) if self.train_args['action']=='elem' else 1.0
                            values = tf.ones(shape=self.action_sample.shape, dtype=tf.float32)
                            E_Q = 0.0
                        else:
                            train_loss, E_Q, action, act_grad, values = self._rl_train_step(data['inputs'], data['labels'])
                            with self.logger.as_default():
                                tf.summary.scalar("E_Q", E_Q, step=self.gloabl_train_step)
                                tf.summary.scalar("action", action, step=self.gloabl_train_step)
                                tf.summary.histogram("values", values, step=self.gloabl_train_step)
                        t.set_postfix(st_loss=train_loss.numpy())
                        
                        # Valid
                        if self.gloabl_train_step % self.valid_args['valid_gap'] == 0:
                            with trange(self.dataloader.info['valid_step'], desc="Valid steps", leave=False) as v:
                                self.mv_loss_fn.reset_states()
                                vv_metrics = []
                                for valid_step in v:
                                    v_data = valid_iter.get_next()
                                    v_loss, v_metrics = self._valid_step(v_data['inputs'], v_data['labels'])
                                    v.set_postfix(sv_loss=v_loss.numpy())
                                    vv_metrics.append(v_metrics)
                                ev_loss = self.mv_loss_fn.result()
                                ev_metric = tf.reduce_mean(v_metrics)
                                
                            self.mem_experience_buffer(weight=self.model.trainable_weights, 
                                                       metric=ev_metric, 
                                                       action=(action, act_grad), 
                                                       values=values,
                                                       E_Q = E_Q,
                                                       step=self.gloabl_train_step)
                        self.gloabl_train_step += 1
                    et_loss = self.mt_loss_fn.result()
                
                # Test
                with trange(self.dataloader.info['test_step'], desc="Test steps") as t:
                    self.mtt_loss_fn.reset_states()
                    tt_metrics = []
                    for test_step in t:
                        t_data = test_iter.get_next()
                        t_loss,t_metric = self._test_step(t_data['inputs'], t_data['labels'])
                        t.set_postfix(test_loss=t_loss.numpy())
                        tt_metrics.append(t_metric)
                    ett_loss = self.mtt_loss_fn.result()
                    ett_metric = tf.reduce_mean(tt_metrics)
                    
                e.set_postfix(et_loss=et_loss.numpy(), ett_metric=ett_metric.numpy(), ett_loss=ett_loss.numpy())
                with self.logger.as_default():
                    tf.summary.scalar("et_loss", et_loss, step=epoch)
                    tf.summary.scalar("ev_metric", ev_metric, step=epoch)
                    tf.summary.scalar("ett_mloss", ett_loss, step=epoch)
                    tf.summary.scalar("ett_metric", ett_metric, step=epoch)
                    
        self.save_experience(q_mode=self.valid_args["q_mode"])
                
                
        