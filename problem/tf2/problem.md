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


## ImportError: cannot import name 'dtensor' from 'tensorflow.compat.v2.experimental' 

在裝`tensorflow`似乎已經安裝`keras`不過版本好像是錯的，所以必須重新安裝對的`keras`版本。因為當初是安裝2.6版，所以也就安裝`keras = 2.6`就可以了。

## pyright無法補`tf.keras`

[參考網站](https://bytemeta.vip/repo/microsoft/pylance-release/issues/1941)

不明原因，`tf.keras`無法進行補字，後來在找了很久的資料，終於找了，必須在`tensorflow`的模組中的`__init__.py`加⼊一些載⼊的程式碼片段，才能順利進行補字。只能說補字這種東西還是不要習慣，不然真的沒有它就不會打程式了，相當困擾。

```python

# Explicitly import lazy-loaded modules to support autocompletion.
# pylint: disable=g-import-not-at-top
if _typing.TYPE_CHECKING:
  from tensorflow_estimator.python.estimator.api._v2 import estimator as estimator
  from keras.api._v2 import keras
  from keras.api._v2.keras import losses
  from keras.api._v2.keras import metrics
  from keras.api._v2.keras import optimizers
  from keras.api._v2.keras import initializers
# pylint: enable=g-import-not-at-top

```




