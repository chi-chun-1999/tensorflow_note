# 線性迴歸模r

[程式碼](../program/simple_linear_regression.py)


在這個範例中，演示了如何使用`tensorflow`+`python`計算線性迴歸模型，裡面包含了很多概念。其中有使用到`tensorflow`所提供的自動求解梯度、feedforward Network，以及演示使用`for`迴圈與`tensorflow`建立多維的差別。


## 自動求解梯度

在`tensorflow`中提供了自動求解梯度的方式，以下程式碼示範如何使用自己的模型，以及損失函數，並使用`tensorflow`的自動求解梯度來計算梯度值：

```python
@tf.function
def stepGradient(current_w, current_b, x_train, y_train, lr):
    with tf.GradientTape() as tape:
        tape.watch([current_w,current_b])
        pred_y = feedForwardNetwork(x_train, current_w, current_b)
        loss = mse(y_train, pred_y)
    
    gradients = tape.gradient(target = loss, sources= [current_w,current_b])
    new_w = current_w - lr*gradients[0]
    new_b = current_b - lr*gradients[1]

    return [new_w,new_b]
```

