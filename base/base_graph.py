import tensorflow as tf

with tf.Graph().as_default() as g1:
    v = tf.compat.v1.get_variable("v",initializer=tf.zeros_initializer())






