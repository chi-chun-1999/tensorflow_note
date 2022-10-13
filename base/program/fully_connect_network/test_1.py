import tensorflow as tf
from tensorflow.keras import layers


##
# implement using tensor
x = tf.random.normal([4,28*28])
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x,w1)+b1
o1 = tf.nn.relu(o1)



##  
# implement using keras
x = tf.random.normal([4,28*28])

fc = layers.Dense(512,activation = tf.nn.relu)

h1 = fc(x)
