import tensorflow as tf
from tensorflow.keras import layers

##
# implement using tensor
x = tf.random.normal([4,28*28])

# first layer
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

# second layer
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

# third layer
w3 = tf.Variable(tf.random.truncated_normal([128,64],stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))

# fourth layer
w4 = tf.Variable(tf.random.truncated_normal([64,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))


with tf.GradientTape() as tape:
    h1 = x@w1 + tf.broadcast_to(b1,[x.shape[0],256])
    h1 = tf.nn.relu(h1)

    h2 = h1@w2 + b2
    h2 = tf.nn.relu(h2)

    h3 = h2@w3 + b3
    h3 = tf.nn.relu(h3)

    h4 = h3@w4 + b4
    h4 = tf.nn.relu(h4)

## 
# implemen using keras
# method 1

fc1 = layers.Dense(256,activation=tf.nn.relu)
fc2 = layers.Dense(128,activation=tf.nn.relu)
fc3 = layers.Dense(64,activation=tf.nn.relu)
fc4 = layers.Dense(10,activation=tf.nn.relu)

h1 = fc1(x)
h2 = fc2(h1)
h3 = fc3(h2)
h4 = fc4(h3)

# method 2

from tensorflow.keras import layers, Sequential

model = Sequential([
        layers.Dense(256,activation=tf.nn.relu),
        layers.Dense(128,activation=tf.nn.relu),
        layers.Dense(64,activation=tf.nn.relu),
        layers.Dense(10,activation=tf.nn.relu),
    ])

out = model(x)



