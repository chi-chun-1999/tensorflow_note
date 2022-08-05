import chun_data.transfer as dt
import numpy as np
import tensorflow as tf



if __name__ == "__main__":

    PROJECT_ROOT = "../../../../"

    data_transfer = dt.DataTrasnfer()

    test_images = data_transfer.idx_to_tensor(PROJECT_ROOT + "dataset/t10k-images-idx3-ubyte")
    test_label = data_transfer.idx_to_tensor(PROJECT_ROOT + "dataset/t10k-labels-idx1-ubyte")

    train_images = data_transfer.idx_to_tensor(PROJECT_ROOT + "dataset/train-images-idx3-ubyte")
    train_label = data_transfer.idx_to_tensor(PROJECT_ROOT + "dataset/train-labels-idx1-ubyte")



    print(tf.shape(test_images))
    print(tf.shape(test_label))
    print(tf.shape(train_images))
    print(tf.shape(train_label))







