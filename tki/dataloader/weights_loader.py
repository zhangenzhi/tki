import os
import random
import tensorflow as tf

from tki.tools.utils import get_yml_content, print_green
from tki.dataloader.utils import glob_tfrecords
from tki.dataloader.base_dataloader import BaseDataLoader


class DNNWeightsLoader(BaseDataLoader):

    def __init__(self, dataloader_args):
        super(DNNWeightsLoader, self).__init__(dataloader_args=dataloader_args)
        self.feature_config, self.info = self._feature_config_parse(self.dataloader_args['path'], 
                                                    name='feature_configs.yaml')
        self.replay_buffer = []
        self.replay_buffer = self._build_replay_buffer()
        
    def _build_replay_buffer(self):
        # Take care of random restart
        replay_buffer = glob_tfrecords(
        self.dataloader_args['path'], glob_pattern='*.tfrecords')
        if len(replay_buffer) > self.dataloader_args["replay_window"]:
            replay_buffer = replay_buffer[:self.dataloader_args["replay_window"]]
            self.info = self.get_info_inference(num_of_students=self.dataloader_args["replay_window"],
                                                sample_per_student=self.info["sample_per_student"])
        return replay_buffer
            
    def get_info_inference(self, num_of_students, sample_per_student):
        total_sample = num_of_students * sample_per_student
        train_samples = int(total_sample*0.6)
        valid_samples =int(total_sample*0.2)
        test_samples = int(total_sample*0.2)
        train_step = int(train_samples / self.dataloader_args['batch_size'])
        valid_step = int(valid_samples / self.dataloader_args['batch_size'])
        test_step = int(test_samples / self.dataloader_args['batch_size'])
        
        info = {'epochs': self.dataloader_args['epochs'],
                'num_of_students': num_of_students,
                'sample_per_student': sample_per_student,
                'total_samples': total_sample,
                'train_samples': train_samples,
                'test_samples': test_samples, 
                'valid_samples': valid_samples,
                'train_step': train_step,
                'valid_step': valid_step,
                'test_step': test_step,
            }
        
        return info
    
    def _feature_config_parse(self, config_path, name='feature_configs.yaml'):
        yaml_path = os.path.join(config_path, name)
        yaml_feature_config = get_yml_content(yaml_path)
        info = self.get_info_inference(yaml_feature_config['num_of_students'],
                                            yaml_feature_config['sample_per_student'])
        
        feature_config = {
                'valid_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'vars': {"type": 'value', "length": 1, "dtype": tf.string},
            }

        return feature_config, info

    def _make_tensor_describs(self, feature_config=None):

        feature_describs = {}
        for feature, info in feature_config.items():
            if info['type'] == 'list':
                for i in range(info["length"]):
                    feature_type = tf.io.FixedLenFeature([], info["dtype"])
                    feature_describs[feature +
                                             "_{}".format(i)] = feature_type
                info_type = tf.io.FixedLenFeature([], tf.int64)
                feature_describs[feature + "_length"] = info_type
            elif info['type'] == 'value':
                feature_type = tf.io.FixedLenFeature([], info["dtype"])
                feature_describs[feature] = feature_type
            else:
                raise ("no such type to describe")
        return feature_describs

    def _load_tensor_from_tfrecord(self, filelist, feature_config):

        raw_analyse_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_analyse_dataset = raw_analyse_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE),
            block_length=256,
            cycle_length=16,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

        analyse_feature_describ = self._make_tensor_describs(
            feature_config = feature_config)

        def _parse_weights_function(example_proto):
            example = tf.io.parse_example(
                example_proto, analyse_feature_describ)
            parsed_example = {}
            for feat, tensor in analyse_feature_describ.items():
                if example[feat].dtype == tf.string:
                    parsed_example[feat] = tf.io.parse_tensor(example[feat], out_type=tf.float32)
                else:
                    parsed_example[feat] = example[feat]

            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_weights_function,
                                                         num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

        parsed_analyse_dataset = parsed_analyse_dataset.prefetch(tf.data.AUTOTUNE)

        return parsed_analyse_dataset
    
    def load_dataset(self, new_students=[]):
        
        print_green("weight_space_path:{}".format(self.dataloader_args['path']))
        filelist = glob_tfrecords(self.dataloader_args['path'], glob_pattern='*.tfrecords')
        
        # random exp
        if self.dataloader_args['exp'] == "random":
            if len(filelist) > self.dataloader_args['replay_window']:
                random.shuffle(filelist)
                filelist = list(set(filelist) - set(new_students))
                past = [filelist.pop() for _ in range(self.dataloader_args['replay_window'] - len(new_students))]
                filelist = new_students + past
            print("filelist length: {}".format(len(filelist)))
        elif self.dataloader_args['exp'] == "decay":
            #decay exp
            if len(self.replay_buffer) > self.dataloader_args['replay_window']:
                for i in range(len(new_students)):
                    self.replay_buffer.pop(0)
                self.replay_buffer += new_students
            else:
                self.replay_buffer += new_students
            filelist = self.replay_buffer
                
        print_green("students:{}".format(filelist))
        self.info = self.get_info_inference(num_of_students=len(filelist),sample_per_student=self.info['sample_per_student'])
        full_dataset = self._load_tensor_from_tfrecord(filelist=filelist, feature_config=self.feature_config)
        full_dataset = full_dataset.shuffle(1000)
        
        train_dataset = full_dataset.take(self.info['train_samples'])
        
        valid_dataset = full_dataset.skip(self.info['train_samples'])
        valid_dataset = valid_dataset.take(self.info['valid_samples'])
        
        test_dataset = valid_dataset.skip(self.info['valid_samples'])
        test_dataset = valid_dataset.take(self.info['test_samples'])
    
        train_dataset = train_dataset.repeat(self.info["epochs"])
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'])
        
        valid_dataset = valid_dataset.repeat(self.info["epochs"])
        valid_dataset = valid_dataset.batch(self.dataloader_args['batch_size'])
        
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size'])
        
        return train_dataset, valid_dataset, test_dataset
    

class DNNSumReduce(DNNWeightsLoader):
    def __init__(self, dataloader_args):
        super(DNNSumReduce, self).__init__(dataloader_args=dataloader_args)

    def _feature_config_parse(self, config_path, name='feature_configs.yaml'):
        yaml_path = os.path.join(config_path, name)
        yaml_feature_config = get_yml_content(yaml_path)
        info = self.get_info_inference(yaml_feature_config['num_of_students'],
                                            yaml_feature_config['sample_per_student'])
        
        feature_config = {
                'valid_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'vars': {"type": 'value', "length": 1, "dtype": tf.string},
            }

        return feature_config, info

    def _make_sumreduce_describs(self, feature_config=None):
        
        feature_describs = {}
        for feature, info in feature_config.items():
            if info['type'] == 'value':
                feature_type = tf.io.FixedLenFeature([], info["dtype"])
                feature_describs[feature] = feature_type
            else:
                raise ("no such type to describe")
        feature_describs["vars_length"] = tf.io.FixedLenFeature([], tf.int64)
        return feature_describs

    def _load_tensor_from_tfrecord(self, filelist, feature_config):

        raw_analyse_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_analyse_dataset = raw_analyse_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE),
            block_length=256,
            cycle_length=16,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

        analyse_feature_describ = self._make_sumreduce_describs(
            feature_config = feature_config)

        def _parse_weights_function(example_proto):
            example = tf.io.parse_example(
                example_proto, analyse_feature_describ)
            parsed_example = {}
            for feat, tensor in analyse_feature_describ.items():
                if example[feat].dtype == tf.string:
                    parsed_example[feat] = tf.io.parse_tensor(example[feat], out_type=tf.float32)
                else:
                    parsed_example[feat] = example[feat]

            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_weights_function,
                                                         num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

        parsed_analyse_dataset = parsed_analyse_dataset.prefetch(tf.data.AUTOTUNE)

        return parsed_analyse_dataset


class DNNRL(DNNWeightsLoader):
    def __init__(self, dataloader_args):
        super(DNNRL, self).__init__(dataloader_args=dataloader_args)

    def _feature_config_parse(self, config_path, name='feature_configs.yaml'):
        yaml_path = os.path.join(config_path, name)
        yaml_feature_config = get_yml_content(yaml_path)
        info = self.get_info_inference(yaml_feature_config['num_of_students'],
                                            yaml_feature_config['sample_per_student'])
        
        # feature_config = {
        #             'states': {"type": 'value', "length": 1, "dtype": tf.string},
        #             'metrics':{"type": 'value', "length": 1, "dtype": tf.string},
        #             'Q':{"type": 'value', "length": 1, "dtype": tf.string},
        #             'actions':{"type": 'value', "length": 1, "dtype": tf.string},
        #             'act_grads':{"type": 'value', "length": 1, "dtype": tf.string},
        #             'steps':{"type": 'value', "length": 1, "dtype": tf.string},
        #             'rewards':{"type": 'value', "length": 1, "dtype": tf.string}
        #     }
        feature_config = {
                'state': {"type": 'value', "length": 1, "dtype": tf.string},
                'reward':{"type": 'value', "length": 1, "dtype": tf.string}
        }

        return feature_config, info

    def _make_rl_describs(self, feature_config=None):
        
        feature_describs = {}
        for feature, info in feature_config.items():
            if info['type'] == 'value':
                feature_type = tf.io.FixedLenFeature([], info["dtype"])
                feature_describs[feature] = feature_type
            else:
                raise ("no such type to describe")
        # feature_describs["vars_length"] = tf.io.FixedLenFeature([], tf.int64)
        return feature_describs

    def _load_tensor_from_tfrecord(self, filelist, feature_config):
        
        # filelist = ["./log/DE-rl-step/weight_space/45.tfrecords"]
        raw_dataset = tf.data.Dataset.from_tensor_slices(filelist)

        raw_dataset = raw_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=32),
            block_length=256,
            cycle_length=32,
            num_parallel_calls=16,
            deterministic=False)

        feature_describ = self._make_rl_describs(feature_config = feature_config)

        def _parse_weights_function(example_proto):
            example = tf.io.parse_example(example_proto, feature_describ)
            parsed_example = {}
            for feat, tensor in feature_describ.items():
                if example[feat].dtype == tf.string:
                    parsed_example[feat] = tf.io.parse_tensor(example[feat], out_type=tf.float32)
                else:
                    parsed_example[feat] = example[feat]

            return parsed_example

        parsed_dataset = raw_dataset.map(_parse_weights_function, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

        parsed_dataset = parsed_dataset.prefetch(tf.data.AUTOTUNE)

        # ForkedPdb().set_trace()
        
        return parsed_dataset
