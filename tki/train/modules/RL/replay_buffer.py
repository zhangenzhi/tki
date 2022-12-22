import os
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

from tki.tools.utils import check_mkdir, save_yaml_contents
from tki.dataloader.utils import glob_tfrecords

class ReplayBuffer(object):
    def __init__(self, buffer_args, log_path, student_id):
        
        self.args = buffer_args
        self.student_id = student_id
        self.q_style = buffer_args.q_style
        self.log_path= log_path
        self.weight_dir, self.writter,self.weight_file = self._build_writter()
        
        self.discount_factor = buffer_args.discount_factor
        self.training_knowledge = edict({'state':[], 
                                         'reward':[], 
                                         'train_loss':[], 
                                         'train_metric':[], 
                                         'valid_loss':[], 
                                         'valid_metric':[], 
                                         'action':[], 
                                         'act_idx':[],
                                         'act_grad':[], 
                                         'expect_q_values':[],
                                         'step':[]})
        self.save_knowledge = buffer_args.save_knowledge
        
    def _build_writter(self):
        weight_dir = os.path.join(self.log_path, "weight_space")
        check_mkdir(weight_dir)
        weight_file = os.path.join(weight_dir, '{}.tfrecords'.format(self.student_id))
        writter = tf.io.TFRecordWriter(weight_file)
        return weight_dir, writter, weight_file
    
    def update_buffer(self, 
                      state, 
                      expect_q_values, 
                      train_loss,
                      train_metric, 
                      action,
                      act_idx, 
                      act_grad, 
                      valid_loss, 
                      valid_metric, 
                      step):
        # state
        self.training_knowledge.state.append(state)
        
        # loss & metrics
        self.training_knowledge.train_metric.append(train_metric)
        self.training_knowledge.train_loss.append(train_loss)
        self.training_knowledge.valid_loss.append(valid_loss)
        self.training_knowledge.valid_metric.append(valid_metric)
        
        # expect q values 
        self.training_knowledge.expect_q_values.append(expect_q_values)
        
        # action
        self.training_knowledge.act_grad.append(act_grad)
        self.training_knowledge.action.append(action)
        self.training_knowledge.act_idx.append(act_idx)
        self.training_knowledge.step.append(step)
        
        return expect_q_values[0][act_idx]

    def save_experience(self):
        
        if self.q_style == "multi_action_temporal_diffrernce":
            self.multi_action_temporal_diffrernce()
        elif self.q_style == "single_action_temporal_diffrernce":
            self.single_action_temporal_diffrernce()
        elif self.q_style == 'single_action':
            self.single_action()
        elif self.q_style == 'multi_action':
            self.multi_action()
        
        self.write_weights_to_tfrecord()
        self.writter.close()
        
    def training_knowledge_example(self):        
        configs = {}
        examples = []

        values = list(self.training_knowledge.values())
        keys = list(self.training_knowledge.keys())
        sample_value = list(zip(*values))
        samples = [dict(zip(keys, s)) for s in sample_value]
        for s in samples:
            feature = {}
            for feature_name, value in s.items():
                if feature_name in self.save_knowledge:
                    bytes_v = tf.io.serialize_tensor(value).numpy()
                    feature[feature_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_v]))
                    configs[feature_name] = {'type': 'bytes', 'shape': value.shape.as_list()}
            examples.append(tf.train.Example(features=tf.train.Features(feature=feature)))

        return examples, configs
    
    def write_weights_to_tfrecord(self):
        
        examples, configs = self.training_knowledge_example()
        for e in examples:
            self.writter.write(e.SerializeToString())
        
        config_path = os.path.join(self.weight_dir, 'feature_configs.yaml')

        configs['num_of_students'] = len(glob_tfrecords(
            self.weight_dir, glob_pattern='*.tfrecords'))
        configs['sample_per_student'] = int(len(self.training_knowledge.state))
        configs['total_samples'] = configs['sample_per_student'] * \
            configs['num_of_students']
        save_yaml_contents(contents=configs, file_path=config_path)

    ## Q value styles
    def single_action(self):
        s = len(self.training_knowledge['rewards'])
        Q = [tf.constant(10.0,shape=self.training_knowledge['rewards'][-1].shape)] 
        for i in reversed(range(s-1)):
            q_value = self.training_knowledge['rewards'][i] + self.discount_factor*Q[0]
            Q.insert(0, q_value)
        self.training_knowledge['Q'] = [v for v in Q]
        with self.logger.as_default():
            for i in range(len(Q)):
                tf.summary.scalar("T_Q", tf.squeeze(Q[i]), step=i)

    def single_action_temporal_diffrernce(self):
        s = len(self.training_knowledge['rewards'])
        Q = []
        for i in range(s):
            q_value = self.training_knowledge['rewards'][i] + df*self.training_knowledge['E_Q'][i]
            Q.append(q_value)
        self.training_knowledge['Q'] = [v for v in Q]
        with self.logger.as_default():
            for i in range(len(Q)):
                tf.summary.scalar("T_Q", tf.squeeze(Q[i]), step=i)

    def multi_action(self):
        expect_q_values = self.training_knowledge.expect_q_values
        valid_metric = self.training_knowledge.valid_metric
        act_idx = self.training_knowledge.act_idx
        
        rewards = [valid_metric[-1]]
        for idx in reversed(range(len(valid_metric))):
            rewards.append(valid_metric[idx] + self.discount_factor*rewards[-1])
        rewards = list(reversed(rewards))
        rewards.pop()
            
        for q, r, a in zip(expect_q_values, rewards, act_idx):
            np_q = q.numpy()
            np_q[0][a] = r
            self.training_knowledge.reward.append(tf.constant(np_q))

    def multi_action_temporal_diffrernce(self):
        expect_q_values = self.training_knowledge.expect_q_values
        valid_metric = self.training_knowledge.valid_metric
        
        expect_q_values.append([0.0])
        for idx in range(len(valid_metric)):
            r = valid_metric[idx] + max(expect_q_values[idx+1]) * self.discount_factor
            self.training_knowledge.reward.append(tf.constant(r))
        expect_q_values.pop()

    def weight_augmentation(self, weights):
        pass
    
#----- augmentation -----
def weights_augmentation(weights, style):
    if style=="flat_imagelization":
        return flat_imagelization(weights=weights)
    else:
        return stack_imagelization(weights=weights)

def flat_imagelization(weights):
    img = []
    for w in weights:
        img.append(tf.reshape(w,shape=(1,-1)))
    img = tf.concat(img, axis=-1)
    return tf.reshape(img, shape=(-1, int(np.sqrt(img.shape[0])), 3))

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