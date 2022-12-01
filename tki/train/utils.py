import os
import sys
import pdb
import tensorflow as tf
from tki.tools.utils import print_warning

def prepare_dirs(valid_args):
    if valid_args.get('log_path'):
        valid_args['log_file'] = os.path.join(valid_args['log_path'], 'log_file.txt')
        valid_args['model_dir'] = os.path.join(valid_args['log_path'], 'models')
        valid_args['tensorboard_dir'] = os.path.join(valid_args['log_path'], 'tensorboard')
        mkdirs(valid_args['model_dir'])
        mkdirs(valid_args['tensorboard_dir'])

        # weights pool
        if valid_args.get('analyse'):
            valid_args['analyse_dir'] = os.path.join(valid_args['log_path'],
                                                     'analyse/{}'.format(valid_args['analyse']['format']))
            if not os.path.isdir(valid_args['analyse_dir']):
                mkdirs(valid_args['analyse_dir'])
            target_model_version = len(os.listdir(valid_args['analyse_dir']))
            valid_args['analyse_dir'] = os.path.join(valid_args['analyse_dir'], str(target_model_version))
            valid_args['log_file'] = os.path.join(valid_args['analyse_dir'], 'log_file.txt')
            mkdirs(valid_args['analyse_dir'])

def check_mkdir(path):
    if not os.path.exists(path=path):
        print_warning("no such path: {}, but we made.".format(path))
        os.makedirs(path)
        
def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_scheduler(name):
    return name

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
            
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

def fix_n_action(self):
    flat_var = self.reduced_space(self.model.trainable_variables)
    state = tf.reshape(tf.concat(flat_var, axis=0), (1,-1))
    self.values = self.supervisor({'state':state})
    return self.action_sample, self.values