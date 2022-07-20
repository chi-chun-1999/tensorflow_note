# Tensorboard

由於要直接讓其他人單純看程式碼想像撰寫出來的模型是不可能的，所以Tensorflow團隊撰寫了一個Tensorboard的工具，讓模型可視化

需要透過FileWriter將要輸出的graph，當作第二個參數才能使用tensorboard輸出

```python
train_writer = tf.compat.v1.summary.FileWriter('/path/to/log', sess.graph)
```


```shell
tensorboard --logdir=<path to log>
```

接著打開瀏覽器的 http://localhost:6006 即可看到輸出結果

