from tqdm import trange
import numpy as np
import tensorflow as tf

# others
from tki.plot.visualization import visualization
from tki.train.student import Student
from tki.tools.utils import print_warning, print_green, print_error, print_normal

class MnistRLStudent(Student):
    
    def __init__(self, student_args, supervisor = None, id = 0):
        super(MnistRLStudent, self).__init__(student_args, supervisor,id)
        
        self.act_idx = []
        self.gloabl_train_step = 0
        self.valid_gap = 100
        
        ## RL
        self.best_metric = 0.5
        
        if self.id < 100:
            self.epsilon = 0.5 
        if self.id >= 100:
            self.epsilon = 0.75
        if self.id >= 300:
            self.epsilon = 0.875
        if self.id >= 700:
            self.epsilon = 0.925
            
        self.action_sample = [0.1,0.5,1.0,2.5,5.0]
        self.index_max = int(len(self.action_sample))
        self.E_Q = 1.0
        self.baseline = 0.1
        self.experience_buffer = {'states':[], 'rewards':[], 'metrics':[], 'actions':[], 'values':[],
                                  'act_grads':[],'E_Q':[], 'steps':[]}
    
    def reduced_space(self, state):
        if self.valid_args['weight_space'] == 'sum_reduce':
            flat_state = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in state], axis=-1)
        elif self.valid_args['weight_space'] == 'first_reduce':
            first_layer = state[:2]
            last_layer = state[2:]
            reduced_grads = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1)) for g in first_layer], axis=-1)
            keep_grads =tf.concat([tf.reshape(g,(1,-1)) for g in last_layer], axis=-1)
            flat_state = tf.concat([reduced_grads,keep_grads], axis=-1)
        elif self.valid_args['weight_space'] == 'norm_reduce':
            flat_state = tf.concat([tf.reshape(tf.reduce_sum(g, axis=-1),(1,-1))/tf.norm(g) for g in state], axis=-1)
        elif self.valid_args['weight_space'] == 'no_reduce':
            flat_state = tf.concat([tf.reshape(g,(1,-1)) for g in state], axis=-1)
        return flat_state
    
    def fix_n_action(self):
        flat_var = self.reduced_space(self.model.trainable_variables)
        state = tf.reshape(tf.concat(flat_var, axis=0), (1,-1))
        self.values = self.supervisor({'state':state})
        return self.action_sample, self.values
    
    def fix_action(self, t_grad):
        action_samples = tf.reshape(tf.constant(self.action_sample, dtype=tf.float32),shape=(-1,1))
        
        state =  self.model.trainable_variables
        next_states = []
        for act in self.action_sample :
            n_s = [s - act*self.optimizer.learning_rate*g for s,g in zip(state, t_grad)]
            next_states.append(n_s)
        reduced_states = [self.reduced_space(s) for s in next_states]
        scaled_states = tf.concat(reduced_states, axis=0)    
        states_actions = {'state':scaled_states}
        self.values = self.supervisor(states_actions)
        return action_samples, self.values
    
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
        self.mt_loss_fn.update_state(t_loss)
                
        # fixed action with pseudo sgd
        if (self.gloabl_train_step %  self.valid_gap )==0:
            if self.train_args['action'] == 'fix':
                _, self.values = self.fix_action(t_grad=t_grad)
            elif self.train_args['action'] == 'fix_n':
                _, self.values = self.fix_n_action()

            # greedy policy
            self.index_max = self.e_greedy_policy(self.values)
            self.E_Q = tf.squeeze(self.values[self.index_max])

        # next state
        act = self.action_sample[self.index_max]
        gradients = [g*act for g in t_grad]
        clip_grads = [tf.clip_by_value(g, clip_value_min=-1.0, clip_value_max=1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(clip_grads, self.model.trainable_variables))
            
        return t_loss, self.E_Q, act, t_grad, self.values
    
    def train(self, supervisor_info=None):
        
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
        
        # load supervisor
        self.supervisor = self._load_supervisor_model(supervisor_info)
        
        total_epochs = self.dataloader.info['epochs']


        # train, valid, write to tfrecords, test
        # tqdm update, logger
        with trange(total_epochs, desc="Epochs") as e:
            for epoch in e:
                
                # lr decay
                if self.train_args["lr_decay"]:
                    if epoch == int(0.5*total_epochs):
                        self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                        print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))
                    elif epoch == int(0.75*total_epochs):
                        self.optimizer.learning_rate = self.optimizer.learning_rate*0.1
                        print("Current decayed learning rate is {}".format(self.optimizer.learning_rate))
                        
                # Train
                with trange(self.dataloader.info['train_step'], desc="Train steps", leave=False) as t:
                    self.mt_loss_fn.reset_states()
                    for train_step in t:
                        data = train_iter.get_next()
                        if self.supervisor == None:
                            train_loss, t_grad, train_metrics = self._train_step(data['inputs'], data['labels'])
                            action = 1.0
                            self.act_idx.append(int(len(self.action_sample)/2))
                            values = tf.ones(shape=(len(self.action_sample)), dtype=tf.float32)
                            E_Q = -1.0
                        else:
                            train_loss, E_Q, action, t_grad, values = self._rl_train_step(data['inputs'], data['labels'])
                        t.set_postfix(st_loss=train_loss.numpy())
                        
                        # Valid && Evaluate
                        if self.gloabl_train_step % self.valid_gap == 0:
                            ev_metric = self.evaluate(valid_iter=valid_iter, E_Q = E_Q, values = values, action=action, t_grad = t_grad)
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
                    tf.summary.scalar("ev_loss", ev_metric, step=epoch)
                    tf.summary.scalar("ett_mloss", ett_loss, step=epoch)
                    tf.summary.scalar("ett_metric", ett_metric, step=epoch)    
                
        self.save_experience(q_mode=self.valid_args["q_mode"])
        self.model.summary()
        self.model_save(name="finished")
        if train_loop_args["visual"]:
            visualization(self.model, 
                          train_iter.get_next(), test_iter.get_next(), 
                          step_size=1e-2, scale=100, 
                          save_to=self.logdir)
        
    def save_experience(self, q_mode="static", df=0.9):
        
        if q_mode == "TD-NQ":
            if self.supervisor == None:
                # baseline without q-net
                values = self.experience_buffer['values']
                self.experience_buffer['Q'] = values
            else:
                # boostrap Q value
                values = self.experience_buffer['values']
                rewards = self.experience_buffer['rewards']
                for i in range(len(rewards)):
                    np_values = values[i].numpy()
                    e_q = rewards[i] + df * values[i][self.act_idx[i]] 
                    np_values[self.act_idx[i]] = e_q
                    values[i] = tf.reshape(tf.constant(np_values), shape=values[i].shape)
                self.experience_buffer['Q'] = values
                
            with self.logger.as_default():
                for i in range(len(values)):
                    tf.summary.scalar("T_Q", tf.squeeze(values[i][self.act_idx[i]]), step=i)

                    
        elif q_mode == "TD":
            s = len(self.experience_buffer['rewards'])
            Q = []
            for i in range(s):
                q_value = self.experience_buffer['rewards'][i] + df*self.experience_buffer['E_Q'][i]
                Q.append(q_value)
            self.experience_buffer['Q'] = [v for v in Q]
            with self.logger.as_default():
                for i in range(len(Q)):
                    tf.summary.scalar("T_Q", tf.squeeze(Q[i]), step=i)
                    
        elif q_mode == 'static':
            s = len(self.experience_buffer['rewards'])
            Q = [tf.constant(10.0,shape=self.experience_buffer['rewards'][-1].shape)] 
            for i in reversed(range(s-1)):
                q_value = self.experience_buffer['rewards'][i] + df*Q[0]
                Q.insert(0, q_value)
            self.experience_buffer['Q'] = [v for v in Q]
            with self.logger.as_default():
                for i in range(len(Q)):
                    tf.summary.scalar("T_Q", tf.squeeze(Q[i]), step=i)
        
        self._write_trail_to_tfrecord(self.experience_buffer)

        return self.best_metric - self.baseline/10 
    
    def mem_experience_buffer(self, weight, metric, action, values=None, E_Q=1.0, step=0):
            
        # state
        reduced_state = self.reduced_space(weight)
        self.experience_buffer['states'].append(reduced_state)
        
        # reward function
        self.experience_buffer['metrics'].append(metric)
        self.experience_buffer['rewards'].append(metric)
        self.experience_buffer['E_Q'].append(tf.cast(E_Q, tf.float32))
        
        # expect state values 
        if values !=None :
            self.experience_buffer['values'].append(values)
        
        # action
        t_grad = action[1]
        reduced_grads = self.reduced_space(t_grad)
        self.experience_buffer['act_grads'].append(reduced_grads)
        self.experience_buffer['actions'].append(tf.constant(action[0]))
        self.experience_buffer['steps'].append(tf.cast(step, tf.float32))
        
        
    def evaluate(self, valid_iter, E_Q, values, action, t_grad):
        
        # warmup sample 
        if self.supervisor == None and self.valid_args["q_mode"] == "TD-NQ":
            raw_values = []
            back_grad = [-action*g for g in t_grad]
            self.optimizer.apply_gradients(zip(back_grad, self.model.trainable_variables))
            
            for i in range(len(self.action_sample)):
                grad = [t_g * self.action_sample[i] for t_g in t_grad]
                self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
                
                v_data = valid_iter.get_next()
                v_loss, v_metrics = self._valid_step(v_data['inputs'], v_data['labels'])
                ev_metric = tf.reduce_mean(v_metrics)
                
                raw_values.append(ev_metric)
                re_grad = [-g for g in grad]
                self.optimizer.apply_gradients(zip(re_grad, self.model.trainable_variables))
            values = tf.concat([10.0 * v for v in raw_values], axis=0)
            
            fore_grad =  [action*g for g in t_grad]
            self.optimizer.apply_gradients(zip(fore_grad, self.model.trainable_variables))
            ev_metric = values[int(len(self.action_sample)/2)]
        else:
            self.mv_loss_fn.reset_states()
            vv_metrics = []
            for valid_step in range(self.dataloader.info['valid_step']):
                v_data = valid_iter.get_next()
                v_loss, v_metrics = self._valid_step(v_data['inputs'], v_data['labels'])
                vv_metrics.append(v_metrics)
            ev_loss = self.mv_loss_fn.result()
            ev_metric = tf.reduce_mean(vv_metrics)
            self.act_idx.append(self.index_max) 
        
        # save sample
        if self.valid_args["q_mode"] == "TD":
            E_Q = 10.0 * ev_metric if E_Q < 0.0 else E_Q
        elif self.valid_args["q_mode"] == "TD-NQ":
            E_Q = ev_metric 
        elif self.valid_args["q_mode"] == "NQ":
            E_Q = 10.0 * ev_metric if E_Q < 0.0 else E_Q
        elif self.valid_args["q_mode"] == "static":
            E_Q = E_Q

        with self.logger.as_default():
            tf.summary.scalar("E_Q", E_Q, step=self.gloabl_train_step)
            tf.summary.scalar("action", action, step=self.gloabl_train_step)
        self.mem_experience_buffer(weight=self.model.trainable_weights, 
                                metric=ev_metric, 
                                action=(action, t_grad), 
                                E_Q = E_Q,
                                values= values,
                                step=self.gloabl_train_step)
        return ev_metric
                
                
        