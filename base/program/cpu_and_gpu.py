import tensorflow as tf
from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np


n=1

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([1,n])
    cpu_b = tf.random.normal([n,1])

with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([1,n])
    gpu_b = tf.random.normal([n,1])


def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a,cpu_b)
        return c

def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a,gpu_b)
        return c

cpu_time = timeit(cpu_run,number=10)
gpu_time = timeit(gpu_run,number=10)

print('wrampu:', cpu_time, gpu_time)

cpu_times = []
gpu_times = []
n_s = []

for i in range(19):
    n=2**i

    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([1,n])
        cpu_b = tf.random.normal([n,1])

    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([1,n])
        gpu_b = tf.random.normal([n,1])

    cpu_time = timeit(cpu_run,number=10)
    gpu_time = timeit(gpu_run,number=10)
    cpu_times.append(cpu_time)
    gpu_times.append(gpu_time)
    n_s.append(n)
    print('run time:', cpu_time, gpu_time)

plt.plot(n_s,cpu_times,'r', label='cpu time')
plt.plot(n_s,gpu_times,'g', label='gpu_time')
plt.legend()
plt.show()




# Example: plot cos(x), sin(x), and sinc(x) in the same figure
#x1 = np.linspace(0, 2*np.pi)    # 50x1 array between 0 and 2*pi
#y1 = np.cos(x1)
#x2 = np.linspace(0, 2*np.pi,20) # 20x1 array
#y2 = np.sin(x2)
#x3 = np.linspace(0, 2*np.pi,10) # 10x1 array
#y3 = np.sinc(x3)
#
#plt.plot(x1, y1, 'k--+')   # black dashed line, with "+" markers
#plt.plot(x2, y2, 'gd')     # green dimonds (no line)
#plt.plot(x3, y3, 'r:')     # red dotted line (no marker)
#plt.show()
