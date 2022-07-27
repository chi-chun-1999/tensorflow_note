import tensorflow as tf


a =  tf.constant([[ 2.,3.],[4.,5.],[6,7]],shape=(3,2),name="a")

b =  tf.constant([[ 4.,4.],[5.,5.],[6,6]],shape=(3,2),name="b")

c = tf.add(a,b)

print(c)
    

d = tf.subtract(a,b)
print(d)


e = tf.multiply(a,b)
print(e)

print(b)
b_t = tf.transpose(b,perm=[1,0])
print(b_t)



f = tf.matmul(a,b_t)
print(f)

g = tf.divide(b,a)
print(g)


h = tf.truediv(b,a)
print(h)


i = tf.math.mod(b,a)
print(i)



a_neg =  tf.constant([[ -2.,3.],[-4.,5.],[-6,7]],shape=(3,2),name="a")
k=tf.abs(a)
print(k)



l = tf.negative(a)
print(l)
