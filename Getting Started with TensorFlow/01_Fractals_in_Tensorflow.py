#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:31:21 2017

Fractals in TensorFlow

@author: Vincent
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X+1j*Y
c = tf.constant(Z.astype(np.complex64))

zs = tf.Variable(c)
ns = tf.Variable(tf.zeros_like(c, tf.float32))

# instantiate an InteractiveSession():
sess = tf.InteractiveSession()
# initialize all the variables involved through the run() method
tf.initialize_all_variables().run()
# Start the iteration:
zs_ = zs*zs + c
#Define the stop condition of the iteration:
not_diverged = tf.abs(zs_) < 4
# use the group operator that groups multiple operations:
step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)))
# run the operator for two hundred step
for i in range(200):
    step.run()                           
# visualize
plt.imshow(ns.eval())
plt.show()
