# %% 
import tensorflow as tf
import abc

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

class DenseLayer(Layer):
    def __init__(self,node_num = 0, activation_function = "relu"):
        self._node_num=0
        self._activation_function=""
        self.setLayer(node_num,activation_function)

    def buildLayer(self,layer_input_num):
        self._layer_input_num = layer_input_num
        self._w = tf.Variable(tf.random.normal(shape=[self._layer_input_num,self._node_num]))
        self._b = tf.Variable(tf.random.normal(shape=[self._node_num],dtype=tf.float32))

    def setLayer(self,node_num,activation_function):
        self._node_num=node_num
        self._activation_function=activation_function

    def getParameter(self):
        return [self._w,self._b]
    
    def getWeight(self):
        return self._w

    def getBias(self):
        return self._b

# %%  


if __name__ == '__main__':

    dense_layer = DenseLayer(node_num=10,activation_function="relu")
    dense_layer.buildLayer(3)

    w , b = dense_layer.getParameter()
    print("w = ",w, "b = ",b)
    


