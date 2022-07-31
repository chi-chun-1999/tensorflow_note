# %% 
import tensorflow as tf 

# %%  
a = tf.constant([[1,2,3,4],[1,2,3,4]])
print(a)
print(tf.shape(a))

# %%  the function of expand dims

print("expand_dim 0 :",tf.expand_dims(a,0))
print("expand_dim 1 :",tf.expand_dims(a,1))
print("expand_dim 2 :",tf.expand_dims(a,2))


# %% 型態轉換
a = tf.constant([1],dtype=tf.int32)
print(a)
a = tf.cast(a,tf.float32)
print(a)

# %%  
b = tf.random.normal(shape=[10,5],dtype=tf.float32)
print(b)

# %%  split data to batch

batch_size = 3
batch_num = int(tf.math.floor(len(b)/batch_size))
#print(b[0:(tf.cast(batch_num,tf.int32)*batch_size)])

split_data = tf.split(b[0:batch_num*batch_size],num_or_size_splits=batch_num,axis=0)
print(tf.shape(split_data))
print(split_data)

# %%  
print(tf.math.floor((len(b)/33)))

# %%  

# %%  

# %%  
