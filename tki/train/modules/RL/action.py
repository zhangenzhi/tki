import numpy as np
import tensorflow as tf

class ActionSpace(object):
    def __init__(self, action_args) -> None:
    
        self.act_space = action_args.act_space
        self.act_style = action_args.style
        self.action_samples = self._expand_action_samples()
    
    def __call__(self, idx):
        return self.get_act_from_idx(idx)
    
    def __len__(self):
        return len(self.action_samples)
    
    # expand samples
    def _expand_action_samples(self):
        action_samples = []
        if self.act_style == "fix_n":
            action_samples = self.act_space
        elif self.act_style == "2_dims":
            for lr in self.act_space.LR:
                for re in self.act_space.RE:
                    action_samples.append([lr, re])
        return action_samples
    
    # getter
    def get_length(self):
        return len(self.action_samples)
    
    def get_act_from_idx(self, idx):
        return self.action_samples[idx] 
    
    def get_random_act(self):
        pass
    
    def predict_values(self, weights, supervisor):
        aug_weights = weights_augmentation(weights)
        values = supervisor({'state':aug_weights})
        return values
    
#----- actions -----
def elem_action(supervisor, model, action_sample, id, t_grad, gloabl_train_step, num_act=1000):
        # fixed action with pseudo sgd
        flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in t_grad]
        flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in model.trainable_variables] 
        flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
        flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))

        if id % 10 == 0:
            action_sample = []
            for g in t_grad:
                shape = g.shape
                action_sample.append( tf.random.uniform(minval=1.0, maxval=1.0, shape=[num_act]+list(shape)))
        else:
            action_sample = []
            for g in t_grad:
                shape = g.shape
                action_sample.append( tf.random.uniform(minval=0.1, maxval=5.0, shape=[num_act]+list(shape)))

        scaled_grads = [g*a for g, a in zip(t_grad, action_sample)]
        flat_scaled_gards = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(num_act, -1)) for g in scaled_grads]
        flat_scaled_gards = tf.concat(flat_scaled_gards, axis=1)
        
        var_copy = tf.tile(flat_var, [flat_scaled_gards.shape.as_list()[0], 1])

        # select wights with best Q-value
        steps = tf.reshape(tf.constant([gloabl_train_step/1000]*num_act, dtype=tf.float32),shape=(-1,1))
        states_actions = {'state':var_copy, 'action':flat_scaled_gards,'step':steps}
        values = supervisor(states_actions)
        return action_sample, values
    
def elem_action(self, t_grad, num_act=1000):
    # fixed action with pseudo sgd
    flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in t_grad]
    flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
    flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
    flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))

    if self.id % 10 == 0:
        self.action_sample = []
        for g in t_grad:
            shape = g.shape
            self.action_sample.append( tf.random.uniform(minval=1.0, maxval=1.0, shape=[num_act]+list(shape)))
    else:
        self.action_sample = []
        for g in t_grad:
            shape = g.shape
            self.action_sample.append( tf.random.uniform(minval=0.1, maxval=5.0, shape=[num_act]+list(shape)))

    scaled_grads = [g*a for g, a in zip(t_grad, self.action_sample)]
    flat_scaled_gards = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(num_act, -1)) for g in scaled_grads]
    flat_scaled_gards = tf.concat(flat_scaled_gards, axis=1)
    
    var_copy = tf.tile(flat_var, [flat_scaled_gards.shape.as_list()[0], 1])

    # select wights with best Q-value
    steps = tf.reshape(tf.constant([self.gloabl_train_step/1000]*num_act, dtype=tf.float32),shape=(-1,1))
    states_actions = {'state':var_copy, 'action':flat_scaled_gards,'step':steps}
    self.values = self.supervisor(states_actions)
    return self.action_sample, self.values

def soft_action(self, t_grad, num_act=64):
    # fixed action with pseudo sgd
    flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in t_grad]
    flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
    flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
    flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))
    if self.id % 10 == 0:
        self.action_sample = tf.random.uniform(minval=1.0, maxval=1.0, shape=(num_act,1))
    else:
        self.action_sample = tf.random.uniform(minval=0.01, maxval=5.0, shape=(num_act,1))
    scaled_gards = flat_grad * self.action_sample
    var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
    # select wights with best Q-value
    steps = tf.reshape(tf.constant([self.gloabl_train_step/10000]*self.action_sample.shape[0], dtype=tf.float32),shape=(-1,1))
    states_actions = {'state':var_copy, 'action':scaled_gards,'step':steps}
    self.values = self.supervisor(states_actions)
    return self.action_sample, self.values

def neg_action(self, t_grad):
    # fixed action with pseudo sgd
    flat_grads = [tf.reshape(tf.math.reduce_sum(g, axis= -1), shape=(-1)) for g in t_grad]
    flat_vars = [tf.reshape(tf.math.reduce_sum(v, axis= -1), shape=(-1)) for v in self.model.trainable_variables] 
    flat_grad = tf.reshape(tf.concat(flat_grads, axis=0), (1,-1))
    flat_var = tf.reshape(tf.concat(flat_vars, axis=0), (1,-1))
    if self.id % 10 == 0:
        self.action_sample = tf.random.uniform(minval=1.0, maxval=1.0, shape=(10,1))
    else:
        self.action_sample = tf.reshape(tf.constant([0.01,0.1,1.0,1.5,2.0,-0.01,-0.1,-1.0,-1.5,-2.0], dtype=tf.float32),shape=(-1,1))
    scaled_gards = flat_grad * self.action_sample
    var_copy = tf.reshape(tf.tile(flat_var, [scaled_gards.shape.as_list()[0], 1]), scaled_gards.shape)
    # select wights with best Q-value
    steps = tf.reshape(tf.constant([self.gloabl_train_step/10000]*self.action_sample.shape[0], dtype=tf.float32),shape=(-1,1))
    states_actions = {'state':var_copy, 'action':scaled_gards,'step':steps}
    self.values = self.supervisor(states_actions)
    return self.action_sample, self.values

def fix_n_action(weights, supervisor):
    aug_weights = weights_augmentation(weights)
    values = supervisor({'state':aug_weights})
    return values

def fix_action(self, t_grad):
    action_samples = tf.reshape(tf.constant(self.action_space, dtype=tf.float32),shape=(-1,1))
    
    state =  self.model.trainable_variables
    next_states = []
    for act in self.action_space :
        n_s = [s - act*self.optimizer.learning_rate*g for s,g in zip(state, t_grad)]
        next_states.append(n_s)
    reduced_states = [self.reduced_space(s) for s in next_states]
    scaled_states = tf.concat(reduced_states, axis=0)    
    states_actions = {'state':scaled_states}
    self.values = self.supervisor(states_actions)
    return action_samples, self.values

#----- augmentation -----
def weights_augmentation(weights, style="flat_imagelization"):
    if style=="flat_imagelization":
        return flat_imagelization(weights=weights)
    else:
        return stack_imagelization(weights=weights)

def flat_imagelization(weights):
    # flatten then to rgb img
    img = []
    for w in weights:
        img.append(tf.reshape(w,shape=(1,-1)))
    img = tf.concat(img, axis=-1)
    img_width = int(np.sqrt(img.shape[-1]/3))+1
    flatten_size = img_width*img_width*3 # rgb img
    img = tf.pad(img, paddings=[[0, 0],[flatten_size-img.shape[-1],0]], mode="CONSTANT")
    img = tf.clip_by_value(img, clip_value_min=-1.0, clip_value_max=1.0)

    return tf.reshape(img, shape=(img_width, img_width, 3))

def stack_imagelization(weights):
    target_shape = weights[0].shape
    img = []
    for w in weights:
        if len(w.shape) == 2:
            img.append(tf.image.resize_with_pad(tf.expand_dims(w,axis=-1),
                                                target_height=target_shape[0],
                                                target_width=target_shape[1]))
        else:
            continue
    return tf.concat(img, axis=-1)