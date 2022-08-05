import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.my_network import MyNetWork
import tensorflow as tf
from src.layer import DenseLayer



if __name__ == '__main__':

    X_data = tf.fill([100,5],value=-0.1)
    y_data = tf.random.normal(shape=[10,1],dtype=tf.float32)
    my_nn = MyNetWork(10)
    my_nn.setLayer([DenseLayer(node_num=3),DenseLayer(3)])
    #my_nn.splitTrainDataToBatches(data)
    my_nn.fit(X_data,y_data)
    #pdb.set_trace()
    my_nn.showAllLayerInfo()

