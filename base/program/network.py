import tensorflow as tf
import pdb


w1 = tf.Variable(tf.random.normal(shape=[2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random.normal(shape=[3,1],stddev=1,seed=1))

@tf.function
def MyNetwork(x):
    a = tf.matmul(x,w1)
    pdb.set_trace()
    y = tf.matmul(a,w2)
    return y


x = tf.constant([2,2],shape=[1,2],dtype=tf.float32)

tf.config.run_functions_eagerly(True)

y = MyNetwork(x)
print(y)



x_2 = tf.constant([3,2],shape=[1,2],dtype=tf.float32)

y = MyNetwork(x_2)


print(y)
tf.config.run_functions_eagerly(False)

x_3 = tf.constant([3,3],shape=[1,2],dtype=tf.float32)

y = MyNetwork(x_3)
print(y)






