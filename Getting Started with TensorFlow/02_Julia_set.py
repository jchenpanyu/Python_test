#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@file: 02_Julia_set.py
@time: 2017/11/6 22:38
author: vincent.chen
contact: vincentchan.sysu@gmail.com
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]
Z = X + 1J*Y

Z = tf.constant(Z.astype("complex64"))

zs = tf.Variable(Z)
ns = tf.Variable(tf.zeros_like(Z, "float32"))

# create our own interactive session:
sess = tf.InteractiveSession()
# initialize the input tensors:
tf.global_variables_initializer().run()

c = complex(0.0,0.75)
zs_ = zs*zs - c

# The grouping operator and the stop iteration's condition will be the same as in the Mandelbrot computation:
not_diverged = tf.abs(zs_) < 4
step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, "float32")))
# Finally, we run the operator for two hundred steps:
for i in range(200):
    step.run()

# visualize the result run the following command:
plt.imshow(ns.eval())
plt.show()
