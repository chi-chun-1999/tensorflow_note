import tensorflow as tf

tf.compat.v1.disable_eager_execution()


a = tf.compat.v1.constant([1,2])
b = tf.compat.v1.constant([3,6])

c = a + b

session = tf.compat.v1.Session()

c_ = session.run(c)
print(c_)





