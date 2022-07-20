# 基本介紹

Tensorflow 2.x 版本中建議使用Eager Execution作爲主要執行的模式，當然舊的Gaph Execution依然存在，也依然能過使用Version 1的API

如果要使用Graph Execution的話，必須禁用`tf.compat.v1.disable_eager_execution()`，才能順利使用Graph Execution。


## 基本運算

在Tensorflow 2.x 中要進行算，必須先將計算圖封裝於函數中，此外還必需使用`@tf.function`的修飾符號。接著呼叫此函數即可進行此運算圖。

[code](./base_add.py)




