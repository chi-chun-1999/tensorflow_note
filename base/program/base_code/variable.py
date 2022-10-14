import tensorflow as tf
import pdb


w1 = tf.Variable(tf.random.normal(shape=[2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random.normal(shape=[3,1],stddev=1,seed=1))

print(w1)
print(w2)



w2.assign([[1],[2],[3]])
print(w2)
