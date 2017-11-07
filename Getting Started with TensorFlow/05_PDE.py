#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@file: 05_PDE.py
@time: 2017/11/7 23:20
author: vincent.chen
contact: vincentchan.sysu@gmail.com
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Let's now see what is the Laplace(U) function and the ancillary functions used:
"""
def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a, dtype=1)

def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]

def laplace(x):
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6., 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)

# First we have to define the dimensions of the problem.
# Let's imagine that our pond is a 500x500 square:
N = 500

# The following two-dimensional tensor is the pond at time t = 0,
# that is, the initial condition of our problem:
u_init = np.zeros([N, N], dtype=np.float32)

# We have 40 random raindrops on it
for n in range(40):
    a, b = np.random.randint(0, N, 2)
    u_init[a, b] = np.random.uniform()

# Using matplotlib, we can show the initial square pond:
#plt.imshow(u_init)
#plt.show()

# Then we define the following tensor:
# It is the temporal evolution of the pond.
# At time t = tend it will contain the final state of the pond.
ut_init = np.zeros([N, N], dtype=np.float32)

# We must define some fundamental parameters and a time step of the simulation:
eps = tf.placeholder(tf.float32, shape=())

# define a physical parameter of the model, namely the damping coefficient:
damping = tf.placeholder(tf.float32, shape=())

# Then we redefine our starting tensors as TensorFlow variables,
# since their values will change over the course of the simulation:
U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Finally, we build our PDE model. It represents the evolution in time of
# the pond after the raindrops have fallen:
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Using the TensorFlow group operator, we define how our pond in time t should evolve:
step = tf.group(U.assign(U_), Ut.assign(Ut_))

# In our session we will see the evolution in time of the pond by 1000 steps,
# where each time step is equal to 0.03s,
# while the damping coefficient is set equal to 0.04.

# create our own interactive session:
sess = tf.InteractiveSession()
# initialize the input tensors:
tf.global_variables_initializer().run()

with sess.as_default():
    for i in range(1000):
        step.run({eps: 0.03, damping: 0.04})
        if i % 50 == 0:
            fig1 = plt.figure()
            fig1.show(U.eval())
            fig1.clear()
            #ax1 = fig1.add_subplot(1, 1, 1)
           # ax1.imshow(U.eval())

