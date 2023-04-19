
import tensorflow as tf
from tqdm import trange

from tki.train.modules.RL.action import weights_augmentation
from tki.train.supervisor import Supervisor

class NaiveSupervisor(Supervisor):
    def __init__(self, supervisor_args, id = 0):
        super(NaiveSupervisor, self).__init__(supervisor_args, id = id)
    
    # def update(self, inputs, labels):
        
    #     with tf.GradientTape() as tape:
    #         predictions = self.model(inputs, training=True)
    #         predictions = tf.squeeze(predictions)
    #         labels = tf.reshape(labels,predictions.shape)
    #         loss = self.loss_fn(labels, predictions)

    #         gradients = tape.gradient(loss, self.model.trainable_variables)

    #         self.optimizer.apply_gradients(
    #             zip(gradients, self.model.trainable_variables))
    #     print(loss)
    
    def update(self, inputs, labels):
        
        with tf.GradientTape() as tape:
            states, act_idx = inputs
            predictions = self.model(states)
            predict_value = tf.gather_nd(params=predictions, indices = tf.reshape(act_idx,(-1,1)),batch_dims=1)
            loss = self.loss_fn(labels, predict_value)
            train_metrics = tf.reduce_mean(self.train_metrics(labels, predict_value))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, gradients, train_metrics
