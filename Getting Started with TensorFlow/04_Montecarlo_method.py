#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@file: 04_Montecarlo_method.py
@time: 2017/11/7 23:11
author: vincent.chen
contact: vincentchan.sysu@gmail.com
"""

import tensorflow as tf
import matplotlib.pyplot as plt

trials = 100
hits = 0

# Generate pseudo-random points inside the square [-1,1]×[-1,1],
#  using the random_uniform function:
x = tf.random_uniform([1],minval=-1, maxval=1, dtype=tf.float32)
y = tf.random_uniform([1],minval=-1, maxval=1, dtype=tf.float32)
pi = []

sess = tf.Session()
"""
Inside the session, we calculate the value of π: the area of the circle is π and that of
the square is 4. The relationship between the numbers inside the circle and the total
of generated points must converge (very slowly) to π, and we count how many
points fall inside the circle equation x2+y2=1.
"""
with sess.as_default():
    for i in range(1,trials):
        for j in range(1,trials):
            if x.eval()**2 + y.eval()**2 < 1 :
                hits = hits + 1
                pi.append((4 * float(hits) / i)/trials)

plt.plot(pi)
plt.show()
