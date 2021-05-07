import tensorflow as tf
from rebyval.dataloader.utils import glob_tfrecords
from rebyval.dataloader.base_dataloader import BaseDataLoader


class DnnWeightsLoader(BaseDataLoader):

    def __init__(self, dataloader_args):
        super(DnnWeightsLoader, self).__init__(dataloader_args=dataloader_args)

    def _make_analyse_describs(self, num_trainable_variables, analyse_feature=None):
        if analyse_feature == None:
            analyse_feature = {
                'train_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'valid_loss': {"type": 'value', "length": 1, "dtype": tf.float32},
                'vars': {"type": 'list', "length": num_trainable_variables, "dtype": tf.string},
            }

        analyse_feature_describs = {}
        for feature, info in analyse_feature.items():
            if info['type'] == 'list':
                for i in range(info["length"]):
                    feature_type = tf.io.FixedLenFeature([], info["dtype"])
                    analyse_feature_describs[feature + "_{}".format(i)] = feature_type
                info_type = tf.io.FixedLenFeature([], tf.int64)
                analyse_feature_describs[feature + "_length"] = info_type
            elif info['type'] == 'value':
                feature_type = tf.io.FixedLenFeature([], info["dtype"])
                analyse_feature_describs[feature] = feature_type
            else:
                print("no such type to describe")
                raise
        return analyse_feature_describs

    def _load_analyse_from_tfrecord(self, filepath, num_trainable_variables):

        filelist = glob_tfrecords(filepath, glob_pattern='*')

        raw_analyse_dataset = tf.data.TFRecordDataset(filelist)

        analyse_feature_describ = self._make_analyse_describs(num_trainable_variables)

        def _parse_analyse_function(example_proto):
            example = tf.io.parse_example(example_proto, analyse_feature_describ)
            parsed_example = {}
            for feat, tensor in analyse_feature_describ.items():
                if example[feat].dtype == tf.string:
                    parsed_example[feat] = tf.io.parse_tensor(example[feat], out_type=tf.float32)
                else:
                    parsed_example[feat] = example[feat]

            return parsed_example

        parsed_analyse_dataset = raw_analyse_dataset.map(_parse_analyse_function)

        return parsed_analyse_dataset

    def load_dataset(self):

        dataset = self._load_analyse_from_tfrecord(filepath=self.dataloader_args['datapath'],
                                                   num_trainable_variables=self.dataloader_args['num_trainable_variables'])
        # import pdb
        # pdb.set_trace()

        full_train_size = self.dataloader_args['num_of_samples']
        valid_size = int(0.2*0.8*full_train_size)
        test_size = int(0.2 * full_train_size)

        test_dataset = dataset.take(test_size)
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size'])


        valid_dataset = dataset.skip(test_size).take(valid_size)
        valid_dataset = valid_dataset.batch(self.dataloader_args['batch_size'])

        train_dataset = dataset.skip(valid_size+test_size)
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'])

        return train_dataset, valid_dataset, test_dataset
