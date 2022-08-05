import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layer import DenseLayer



if __name__ == '__main__':

    dense_layer = DenseLayer(node_num=10,activation_function="relu")
    dense_layer.buildLayer(3)

    w , b = dense_layer.getParameter()
    print("w = ",w, "b = ",b)
    shape = dense_layer.getShape()
    print("shape = ",shape)
        
