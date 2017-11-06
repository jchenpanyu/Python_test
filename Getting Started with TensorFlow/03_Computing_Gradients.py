#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@file: 03_Computing_Gradients.py
@time: 2017/11/6 22:38
author: vincent.chen
contact: vincentchan.sysu@gmail.com
"""

import tensorflow as tf

x = tf.placeholder(tf.float32)
y = 2*x*x

var_grad = tf.gradients(y, x)

# To evaluate the gradient, we must build a session:
with tf.Session() as session:
    # The gradient will be evaluated on the variable x=1:
    var_grad_val = session.run(var_grad, feed_dict={x: 1})
    # var_grad_val value is the feed result, to be printed:
    print(var_grad_val)
