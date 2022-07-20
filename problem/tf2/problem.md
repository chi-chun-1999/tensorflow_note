# Tensorflow 2.x 的問題記錄

## 將`tf.function`轉成`tf.Graph`

[Tensorflow的官方文件](https://www.tensorflow.org/guide/intro_to_graphs#converting_python_functions_to_graphs)

在使用`tf.summary.graph`時發現輸入參數爲`tf.Graph`。在2.x的版本中可以用`<tf.function name>.get_concrete_function(tf.constant(1)).graph.as_graph_def()` 將 `tf.function`轉成`tf.Graph`

```python

@tf.function
def SimpleNetwork(a,b):
    c=tf.multiply(a,b,name="mul_c")
    b=tf.add(a,b,name="add_d")
    e = tf.add(c,b,name="add_e")
    return e

with writer.as_default():
    tf.summary.graph(SimpleNetwork.get_concrete_function(input_1,input_2).graph.as_graph_def())

```

