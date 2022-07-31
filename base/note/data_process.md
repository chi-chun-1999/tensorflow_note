# 資料處理

在人工智慧的領域一定少不了資料上的處理，有時候可能是shape改變，有時候也可能是資料上的切割，又或者是型態上的轉換。在`tensorflow`提供了許多函式庫，讓我們可以進行資料上的處理。


[程式碼](../program/data_process.py)


## 增加維度

在Tensorflow中，想增加維度可以使用`tf.expand_dims`來實現。

```python
a = tf.constant([[1,2,3,4],[1,2,3,4]])
print(a)
print(tf.shape(a))

print("expand_dim 0 :",tf.expand_dims(a,0))
print("expand_dim 1 :",tf.expand_dims(a,1))
print("expand_dim 2 :",tf.expand_dims(a,2))
----

tf.Tensor(
[[1 2 3 4]
 [1 2 3 4]], shape=(2, 4), dtype=int32)
tf.Tensor([2 4], shape=(2,), dtype=int32)
expand_dim 0 : tf.Tensor(
[[[1 2 3 4]
  [1 2 3 4]]], shape=(1, 2, 4), dtype=int32)
expand_dim 1 : tf.Tensor(
[[[1 2 3 4]]

 [[1 2 3 4]]], shape=(2, 1, 4), dtype=int32)
expand_dim 2 : tf.Tensor(
[[[1]
  [2]
  [3]
  [4]]

 [[1]
  [2]
  [3]
  [4]]], shape=(2, 4, 1), dtype=int32)

```

## 型態轉換


型態轉換可以使用`tf.cast`來實現。


```python
a = tf.constant([1],dtype=tf.int32)
print(a)
a = tf.cast(a,tf.float32)
print(a)
--
tf.Tensor([1], shape=(1,), dtype=int32)
tf.Tensor([1.], shape=(1,), dtype=float32)
```

## Split dataset to batches

在Tensorflow中，其實已經有提供類似的功能，不過為了練習如何實現它，所以稍為寫了一下。

```python
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
--
ERROR! Session/line number was not unique in database. History logging moved to new session 196
tf.Tensor(
[[-0.46315444  0.61616397  1.5658962  -0.14649662  2.1237888 ]
 [ 0.56585085 -1.050642   -0.47176996  0.91906327  0.32549778]
 [-0.23738347 -0.07164386 -1.2270422  -1.2622882   0.76639664]
 [ 1.3324385   0.8359842  -0.63867915  0.20055619 -1.5842096 ]
 [ 0.8762088   0.17820221  0.6079121  -1.514595    0.40953127]
 [-0.2161335   1.2653699   0.3513314  -1.2814856   0.39871392]
 [ 0.9106425   0.69657373  0.20510626 -1.4758267   0.7562733 ]
 [ 0.04431542 -0.13034588  1.429623    0.55054814 -0.42086756]
 [-0.86324227  0.03972811 -0.92524713  0.16231184  0.1077223 ]
 [ 2.3988492   0.32315135 -1.0978124  -1.3317274   0.37385577]], shape=(10, 5), dtype=float32)
tf.Tensor([3 3 5], shape=(3,), dtype=int32)
[<tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[-0.46315444,  0.61616397,  1.5658962 , -0.14649662,  2.1237888 ],
       [ 0.56585085, -1.050642  , -0.47176996,  0.91906327,  0.32549778],
       [-0.23738347, -0.07164386, -1.2270422 , -1.2622882 ,  0.76639664]],
      dtype=float32)>, <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[ 1.3324385 ,  0.8359842 , -0.63867915,  0.20055619, -1.5842096 ],
       [ 0.8762088 ,  0.17820221,  0.6079121 , -1.514595  ,  0.40953127],
       [-0.2161335 ,  1.2653699 ,  0.3513314 , -1.2814856 ,  0.39871392]],
      dtype=float32)>, <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
array([[ 0.9106425 ,  0.69657373,  0.20510626, -1.4758267 ,  0.7562733 ],
       [ 0.04431542, -0.13034588,  1.429623  ,  0.55054814, -0.42086756],
       [-0.86324227,  0.03972811, -0.92524713,  0.16231184,  0.1077223 ]],
      dtype=float32)>]
```

## `tf.transpose` and `tf.reshape`

在許多時候，會需要用到shape的轉換，tensorflow提供了兩個不同的方式，且兩種方式都代表不同的涵意。`tf.transpose`比較像是轉罝矩陣，`tf.reshape`則是將原本的矩陣按照原本的順序進行轉換。


[tf.transpose](https://www.tensorflow.org/api_docs/python/tf/transpose)

[tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape)

