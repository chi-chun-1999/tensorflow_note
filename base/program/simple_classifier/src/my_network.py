import tensorflow as tf
import abc



class MyNetWork:
    def __init__(self):
        self._parameter=[]
        self._sequential=[]
    def denseLayer(self,node_num,activationfunction='relu'):
        weight=tf.Variable(tf.random.normal(shape=[-1,node_num],dtype=tf.float32))
        bias=tf.Variable(tf.random.normal(shape=[node_num],dtype=tf.float32))

    def fit(self,x_train, y_trian):
        return 

    def sliptTrainData(self):


        
        











