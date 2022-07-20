import tensorflow as tf

a = tf.constant([[3,2],[3,2]], name="a")
b = tf.constant([[2,3],[3,2]], name="b")

result=a*b
s1=tf.compat.v1.Session()
tf.print(result)






