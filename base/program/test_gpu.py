import tensorflow as tf

print("\x1b[31mthe version: ",tf.__version__,"\x1b[0m")

print("\x1b[31mIs the GPU available? ",tf.test.is_gpu_available())
