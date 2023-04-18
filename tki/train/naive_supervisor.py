
import tensorflow as tf
from tqdm import trange

from tki.train.modules.RL.action import weights_augmentation
from tki.train.supervisor import Supervisor

class NaiveSupervisor(Supervisor):
    def __init__(self, supervisor_args, id = 0):
        super(NaiveSupervisor, self).__init__(supervisor_args, id = id)
    
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
