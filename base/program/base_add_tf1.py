import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def MyAdd(p1,p2):
    outcome  = tf.add(p1,p2,name='MyAdd')
    return outcome


a = tf.compat.v1.constant([5,2])
b = tf.compat.v1.constant([3,-6])

c = MyAdd(a,b)


with tf.compat.v1.Session() as sess:
    c_ = sess.run(c)
    print(c_)






