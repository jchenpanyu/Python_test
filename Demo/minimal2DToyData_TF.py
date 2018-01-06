#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@file: minimal2DToyData_TF.py
@time: 2018/1/5 21:35
author: vincent.chen
contact: vincentchan.sysu@gmail.com
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# The toy spiral data consists of three classes (blue, red, yellow) that are not linearly separable.
N = 100 # number of points per class
D = 2 # dimensionality
K = 1 # number of output
num_class = 3
X = np.zeros((N*num_class, D)) # data matrix (each row = single example)
y = np.zeros(N*num_class, dtype='uint8') # class labels
for j in range(num_class):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

# start tf session
sess = tf.Session()
# set random seed
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# create a 80-20 train-test split
train_percentage = 0.8
train_indices = np.random.choice(len(X), round(len(X)*train_percentage), replace=False)
test_indices = np.array(list(set(range(len(X))) - set(train_indices)))
x_vals_train = X[train_indices]
x_vals_test = X[test_indices]
y_vals_train = y[train_indices]
y_vals_test = y[test_indices]

# declare the batch size and placeholders for the data and target:
batch_size = 50
x_data = tf.placeholder(shape=[None, D], dtype=tf.float64)
y_target = tf.placeholder(shape=[None, K], dtype=tf.float64)

# declare our model variables with the appropriate shape.
hidden_layer_nodes = 300
W_1 = tf.Variable(tf.random_normal(shape=[D, hidden_layer_nodes], dtype=tf.float64), dtype=tf.float64)
b_1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes], dtype=tf.float64), dtype=tf.float64)
W_2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, K], dtype=tf.float64), dtype=tf.float64)
b_2 = tf.Variable(tf.random_normal(shape=[K], dtype=tf.float64), dtype=tf.float64)

# creating the hidden layer output and the second will be creating the final output of the model:
hidden_layer_output = tf.nn.relu(tf.add(tf.matmul(x_data, W_1), b_1))
final_output = (tf.add(tf.matmul(hidden_layer_output, W_2), b_2))

# loss function
loss = tf.reduce_mean(tf.square(y_target - final_output)) # L2 loss

# declare our optimizing algorithm and initialize our variables
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# initialize two lists that we can store our train and test loss.
# In every loop we also want to randomly select a batch from the training data for fitting to the model:
# First we initialize the loss vectors for storage.
loss_vec = []
test_loss = []
for i in range(1000):
    # First we select a random set of indices for the batch.
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # We then select the training values
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    # Now we run the training step
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    # We save the training loss
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))
    # Finally, we run the test-set loss and save it.
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)%50==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

def label_predict(data, W_1, b_1, W_2, b_2):
    hidden_layer_output = tf.nn.relu(tf.add(tf.matmul(data, W_1), b_1))
    final_output = (tf.add(tf.matmul(hidden_layer_output, W_2), b_2))
    return sess.run(final_output)

def accuray(predict, groundtrue):
    predict = np.round(predict).astype(np.int8)
    return np.sum(predict.flatten() == groundtrue.flatten())/len(predict)

pre_train = label_predict(x_vals_train, W_1, b_1, W_2, b_2)
pre_test = label_predict(x_vals_test, W_1, b_1, W_2, b_2)
accuray_train = accuray(pre_train, y_vals_train)
accuray_test = accuray(pre_test, y_vals_test)

print('train accuracy=', accuray_train)
print('test accuracy=', accuray_test)
