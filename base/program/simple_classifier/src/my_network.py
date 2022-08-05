#%load_ext autoreload
#%autoreload 2
import tensorflow as tf
import pdb
#from layer import *
from . import layer

import keras
import importlib


class MyNetWork:
    def __init__(self,batch_size=-1,epoch=100):
        self._parameter=[]
        self._sequential=[]
        self._batch_size=batch_size
        self.epoch = 100

    def setLayer(self,sequential_layer):
        self._sequential = sequential_layer


    def fit(self,x_train, y_trian):
        self._splitTrainDataToBatches(x_train=x_train)
        self._y_train = y_trian

        self._buildAllLayer()

        for i in range(self.batch_num):
            predict_y = self._feedForward(self._batches_data[i])
            print(predict_y)

    def _buildAllLayer(self):
        batches_data_shape = self._batchesDataShape()
        input_layer_ouput_num = batches_data_shape[len(batches_data_shape)-1]
        self._sequential[0].buildLayer(input_layer_ouput_num)
        for i in range(len(self._sequential)-1):
            layer_shape = self._sequential[i].getShape()
            layer_ouput_num = layer_shape[len(layer_shape)-1]
            self._sequential[i+1].buildLayer(layer_ouput_num)

    def showAllLayerInfo(self):
        for i in range(len(self._sequential)):
            print("==========Layer ",i,"=========")
            print("shape : ", self._sequential[i].getShape())
            print("weight: ",self._sequential[i].getWeight())
            print("bias: ",self._sequential[i].getBias())

    def getBatchesData(self):
        return self._batches_data

    def _splitTrainDataToBatches(self,x_train):
        if(self._batch_size==-1):
            self._batch_size=len(x_train)

        self.batch_num = int(tf.math.floor(len(x_train)/self._batch_size))
        self._batches_data = tf.split(x_train[0:self.batch_num*self._batch_size],num_or_size_splits=self.batch_num,axis=0)

    def _batchesDataShape(self):
        return tf.shape(self._batches_data).numpy()
    
    @tf.function
    def _feedForward(self,batch_data):
        tmp=tf.matmul(batch_data,self._sequential[0].getWeight())
        predict_y = tf.add(tmp,self._sequential[0].getBias())

        if self._sequential[0].getActivationFunction() == "relu":
            predict_y = tf.keras.activations.relu(predict_y)
        

        for i in range(1,len(self._sequential)):
            tmp = tf.matmul(predict_y,self._sequential[i].getWeight())
            predict_y = tf.add(tmp,self._sequential[i].getBias())
            if self._sequential[i].getActivationFunction() == "relu":
                predict_y = tf.keras.activations.relu(predict_y)

        return predict_y
            




#X_data = tf.random.normal(shape=[10,5],dtype=tf.float32)
#X_data = tf.ones(shape=[100,5],dtype=tf.float32)
X_data = tf.fill([100,5],value=2.1)
y_data = tf.random.normal(shape=[10,1],dtype=tf.float32)
my_nn = MyNetWork(10)
my_nn.setLayer([layer.DenseLayer(node_num=3),layer.DenseLayer(3)])
#my_nn.splitTrainDataToBatches(data)
my_nn.fit(X_data,y_data)
#pdb.set_trace()
my_nn.showAllLayerInfo()

# %%  



