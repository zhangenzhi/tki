import tensorflow as tf

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