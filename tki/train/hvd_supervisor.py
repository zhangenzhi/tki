
import os
import tensorflow as tf
from datetime import datetime
import horovod.tensorflow as hvd

from tki.tools.utils import check_mkdir
from tki.train.modules.RL.action import weights_augmentation
from tki.train.supervisor import Supervisor

class HVDSupervisor(Supervisor):
    def __init__(self, supervisor_args, id = 0):
        super(HVDSupervisor, self).__init__(supervisor_args, id = id)
    
    def update(self, inputs, labels):
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            predictions = tf.squeeze(predictions)
            labels = tf.reshape(labels,predictions.shape)
            loss = self.loss_fn(labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        print(loss)
    
    def _create_logdir(self):
        logdir = "tensorboard/" + "{}-{}".format(self.name, self.id) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(self.args.log_path, logdir)
        if hvd.rank() == 0:
            check_mkdir(logdir)
        return logdir
