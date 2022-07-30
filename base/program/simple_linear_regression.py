# %% 
from numpy.lib.function_base import gradient
import tensorflow as tf
import plotext as plt
#import matplotlib.pyplot as plt
import numpy as np



# %%  

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

# %% 
def gradientDecent(init_w, init_b, x_train, y_train, lr, num_iterations):

    w = init_w
    b = init_b

    for step in range(num_iterations):
        w, b = stepGradient(w,b,x_train,y_train,lr)
        pred_y = feedForwardNetwork(x_train, w, b)
        loss = mse(y_train, pred_y)
        if step % 50 == 0:
            print(f"iteration:{step}, loss:{loss}, weight:{w}, bias:{b}")

    return [w,b]
    



# %% 
def produceDataset(data,number):
    """produce random dataset 

    :arg1: the empty data
    :arg2: the number of the data
    :returns: dataset

    """
    weight = 432.23324
    bais = 199.42323
    
    #for i in range(number):
    #    x = tf.random.uniform(shape=(),minval=-10,maxval=10,dtype=tf.float64)
    #    nois = tf.random.normal(shape=(),dtype=tf.float64)
    #    y = weight*x+bais+nois
    #    data.append([x,y])
    x = tf.random.uniform(shape=(number,1),minval=-10,maxval=10,dtype=tf.float64)
    nois = tf.random.normal(shape=(number,1),dtype=tf.float64)
    y = weight*x+bais+nois
    data=[x,y]

    return data

# %% 
@tf.function
def mse(y,pred_y):
    sub_tmp = tf.subtract(y,pred_y)
    pow_sub = tf.math.pow(sub_tmp,2)
    sum_pow = tf.reduce_sum(pow_sub)
    mse_value = sum_pow/len(y)
    return mse_value

# %%  
@tf.function
def feedForwardNetwork(x_data,weight,bias):
    """feedForwardNetwork

    :x_data: x for trainning data
    :weight: weith
    :bias: bias

    """
    tmp = tf.matmul(x_data,weight)
    #print("tmp = ", tmp)
    pred_y = tf.add(tmp,bias)
    #print("pred_y = ", pred_y)
    
    return pred_y

# %% 


if __name__ == '__main__':
    data = []
    point_number = 100000
    lr = 0.001
    num_iterations = 10000



    data=produceDataset(data,point_number)
    print(tf.shape(data).numpy())
    #data=tf.transpose(data)
    X = tf.reshape(data[0],shape=(point_number,1))
    y = tf.reshape(data[1],shape=(point_number,1))
    print(tf.shape(X).numpy())
    print(tf.shape(y).numpy())

    print(X)
    print(y)

    init_w = tf.Variable(tf.random.normal(shape=(1,1),dtype=tf.float64))
    init_b = tf.Variable(tf.random.normal(shape=(1,1),dtype=tf.float64))
    print(init_w)
    print(init_b)
    
    w,b = gradientDecent(init_w,init_b,X,y,lr,num_iterations)
    print(w, b)


    #pred_y = feedForwardNetwork(X,w,b)

    #mse_value = mse(y,pred_y)
    #print(mse_value)
    #print("pred_y = ", pred_y)
    
    #plt.plot(data._)
    #plt.show()
    #print(data[1])
    #plt.clf()
    #plt.scatter(data[0],data[1])
    #plt.show()
    #print(data)


# %%  

