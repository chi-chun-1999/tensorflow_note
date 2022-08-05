# %% 
import tensorflow as tf
import abc
#from simple_classifier.deps.test  import hello
import deps.test  

# %% 

class Layer(abc.ABC):
    """Network layer abstract class"""
    def __init__(self):
        self._node_num=0
        self._activation_function=""

    @abc.abstractmethod 
    def buildLayer(self,layer_input_num):
        return NotImplemented

    @abc.abstractmethod
    def setLayer(self,node_num,activation_function):
        self._node_num=node_num
        self._activation_function=activation_function

    @abc.abstractmethod    
    def getParameter(self):
        return NotImplemented

    def getNodeNum(self):
        return self._node_num

    def getActivationFunction(self):
        return self._activation_function

class DenseLayer(Layer):
    def __init__(self,node_num = 0, activation_function = "relu"):
        self._node_num=0
        self._activation_function=""
        self.setLayer(node_num,activation_function)

    def buildLayer(self,layer_input_num):
        self._layer_input_num = layer_input_num
        self._w = tf.Variable(tf.random.normal(shape=[self._layer_input_num,self._node_num]))
        self._b = tf.Variable(tf.random.normal(shape=[self._node_num],dtype=tf.float32))
        #self._w = tf.Variable(tf.ones(shape=[self._layer_input_num,self._node_num]))
        #self._b = tf.Variable(tf.ones(shape=[self._node_num],dtype=tf.float32))
        #self._b = tf.Variable(tf.fill([self._node_num],value=4.2))

    def setLayer(self,node_num,activation_function):
        self._node_num=node_num
        self._activation_function=activation_function

    def getParameter(self):
        return [self._w,self._b]
    
    def getWeight(self):
        return self._w

    def getBias(self):
        return self._b

    def getShape(self):
        return tf.shape(self._w).numpy()

# %%  


