import tensorflow as tf
from tensorflow import keras
from tki.train.utils import ForkedPdb


class Linear(keras.layers.Layer):
    def __init__(self, units=32, seed=100000):

        super(Linear, self).__init__()
        self.seed = seed
        self.units = units


    def build(self, input_shape):
      
        w_init = tf.random_normal_initializer(seed=self.seed)(
            shape=(input_shape[-1], self.units), dtype="float32")
        b_init = tf.zeros_initializer()(shape=(self.units,), dtype="float32")

        self.w = tf.Variable(
            initial_value=w_init,
            trainable=True, name="w",
            constraint=lambda z: tf.clip_by_value(z, -10.0, 10.0)
        )
        self.b = tf.Variable(
            initial_value=b_init, trainable=True,
            name="b",
            constraint=lambda z: tf.clip_by_value(z, -10.0, 10.0)
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        return outputs


class DNN(tf.keras.Model):
    def __init__(self,
                 units=[64, 32, 16, 1],
                 activations=['tanh', 'tanh', 'tanh', 'tanh'],
                 use_bn=False,
                 embedding=False,
                 normalization=False,
                 downsampling=False,
                 seed=100000,
                 **kwargs
                 ):
        super(DNN, self).__init__()

        self.units = units
        self.activations = activations
        self.seed = seed
        self.use_bn = use_bn
        self.downsampling = downsampling
        self.normalization = normalization
        
        self.flatten = tf.keras.layers.Flatten()
        self.ds = self._build_downsmapling()
        self.nl = self._build_normalization()
        self.fc_layers = self._build_fc()
        self.fc_act = self._build_act()
        self.fc_bn = self._build_bn()

    def _build_fc(self):
        layers = []
        for units in self.units:
            layers.append(Linear(units=units, seed=self.seed))
        return layers

    def _build_downsmapling(self):
        if self.downsampling:
            ds = tf.keras.layers.Conv2D(filters=3,kernel_size=(3,3),strides=8)
            return ds
    
    def _build_normalization(self):
        if self.normalization:
            nl = tf.keras.layers.BatchNormalization()
            return nl
            
    def _build_bn(self):
        bn = []
        if self.use_bn:
            for _ in range(len(self.activations)-1):
                bn.append(tf.keras.layers.BatchNormalization())
            bn.append(tf.keras.layers.Lambda(lambda x: x))
        return bn

    def _build_act(self):
        acts = []
        for act in self.activations:
            acts.append(tf.keras.layers.Activation(act))
        return acts

    def call(self, inputs):
        
        x = inputs
        
        if self.normalization:
            x = self.nl(x)
            
        if self.downsampling:
            x = self.ds(x)
            x = self.flatten(x)
        else:
            x = self.flatten(x)
            
        if self.use_bn:
            for layer, act, bn in zip(self.fc_layers, self.fc_act, self.fc_bn):
                x = layer(x)
                x = act(x)
                x = bn(x)
        else:
            for layer, act in zip(self.fc_layers, self.fc_act):
                x = layer(x)
                x = act(x)
        return x
