import tensorflow as tf


@tf.function
def SimpleNetwork(a,b):
    
    c=tf.multiply(a,b,name="mul_c")
    b=tf.add(a,b,name="add_d")
    e = tf.add(c,b,name="add_e")

    return e

input_1 = tf.constant([2,3],name="input_1")
input_2 = tf.constant([3,7],name="input_2")

tf.print(SimpleNetwork(input_1,input_2))


