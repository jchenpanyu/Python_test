# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 12
Page: 351-365
@author: vincchen
"""

"""
######    Classifying handwritten digits    ######
"""


"""
Obtaining the MNIST dataset
The MNIST dataset is publicly available at http://yann.lecun.com/exdb/mnist/ and
consists of the following four parts:
• Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, and 60,000 samples)
• Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, and 60,000 labels)
• Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, unzipped and 10,000 samples)
• Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, and 10,000 labels)

The images are stored in byte format, and we will read them into NumPy arrays that we will 
use to train and test our MLP implementation:
"""

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels
"""
The load_mnist function returns two arrays, the first being an nm× dimensional NumPy array (images), 
where n is the number of samples and m is the number of features. The training dataset consists of 
60,000 training digits and the test set contains 10,000 samples, respectively. The images in the 
MNIST dataset consist of 2828× pixels, and each pixel is represented by a gray scale intensity value. 
Here, we unroll the 2828× pixels into 1D row vectors, which represent the rows in our image array 
(784 per row or image). The second array (labels) returned by the load_mnist function contains 
the corresponding target variable, the class labels (integers 0-9) of the handwritten digits.
"""


"""
By executing the following code, we will now load the 60,000 training instances as well as 
the 10,000 test samples from the mnist directory where we unzipped the MNIST dataset:
"""
X_train, y_train = load_mnist('mnist', kind='train')
X_test, y_test = load_mnist('mnist', kind='t10k')

print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


"""
To get a idea what the images in MNIST look like, let's visualize examples of the digits 0-9 after 
reshaping the 784-pixel vectors from our feature matrix into the original 28 × 28 image that 
we can plot via matplotlib's imshow function:
"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()

"""
In addition, let's also plot multiple examples of the same digit to see how different 
those handwriting examples really are:
"""
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

"""
If we decide to save those CSV files, we can execute the following code in our Python session after
loading the MNIST data into NumPy arrays:

np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')

Once we have saved the CSV files, we can load them back into Python using NumPy's genfromtxt function:
    
X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
y_train = np.genfromtxt('train_labels.csv',dtype=int, delimiter=',')
X_test = np.genfromtxt('test_img.csv', dtype=int, delimiter=',')
y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')
"""


"""
Implementing a multi-layer perceptron

In this subsection, we will now implement the code of an MLP with one input, one hidden, and 
one output layer to classify the images in the MNIST dataset.
"""
from scipy.special import expit
import sys

class NeuralNetMLP(object):
    def __init__(self, n_output, n_features, n_hidden=30, 
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        # expit is equivalent to 1.0/(1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1]+1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0]+1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))
    
    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())
        
    def _get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        # backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))
        return grad1, grad2

    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)
        for i in range(self.epochs):
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const*i)
            
            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()
            
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]


            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)
                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)
                # update weights
                delta_w1, delta_w2 = self.eta * grad1,self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return self
            
                
"""
Now, let's initialize a new 784-50-10 MLP,
a neural network with 784 input units (n_features), 50 hidden units (n_hidden), and 10 output units (n_output):
"""
nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50, 
                  l2=0.1, l1=0.0, epochs=1000, eta=0.001, alpha=0.001, decrease_const=0.00001, 
                  shuffle=True, minibatches=50, random_state=1)
"""
As you may have noticed, by going over our preceding MLP implementation,
we also implemented some additional features, which are summarized here:
* l2: The lamda parameter for L2 regularization to decrease the degree of overfitting; equivalently,
      l1 is the lamda parameter for L1 regularization.
* epochs: The number of passes over the training set.
* eta: The learning rate.
* alpha: A parameter for momentum learning to add a factor of the previous gradient to the weight 
         update for faster learning
* decrease_const: The decrease constant d for an adaptive learning rate n that decreases over time for better convergence
* shuffle: Shuffling the training set prior to every epoch to prevent the algorithm from getting stuck in cycles.
* Minibatches: Splitting of the training data into k mini-batches in each epoch.
                The gradient is computed for each mini-batch separately instead of 
                the entire training data for faster learning.
"""

"""
Next, we train the MLP using 60,000 samples from the already shuffled MNIST training dataset.
Before you execute the following code, please note that training the neural network may take
10-30 minutes on standard desktop computer hardware:"""
nn.fit(X_train, y_train, print_progress=True)


"""
Similar to our previous Adaline implementation, we save the cost for each epoch
in a cost_ list that we can now visualize, making sure that the optimization algorithm 
reached convergence. Here, we only plot every 50th step to account for
the 50 mini-batches (50 mini-batches × 1000 epochs). The code is as follows:
"""
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()
"""
As we see in the following plot, the graph of the cost function looks very noisy.
This is due to the fact that we trained our neural network with mini-batch learning,
a variant of stochastic gradient descent.
"""


"""
Although we can already see in the plot that the optimization algorithm converged after 
approximately 800 epochs (40,000/50 = 800), let's plot a smoother version of the cost function against 
the number of epochs by averaging over the mini-batch intervals. The code is as follows:
"""
batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()


"""
Now, let's evaluate the performance of the model by calculating the prediction accuracy:
"""
y_train_pred = nn.predict(X_train)
acc = float(np.sum(y_train == y_train_pred, axis=0)) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))
"""
As we can see, the model classifies most of the training digits correctly, 
but how does it generalize to data that it has not seen before?
Let's calculate the accuracy on 10,000 images in the test dataset:
"""
y_test_pred = nn.predict(X_test)
acc = float(np.sum(y_test == y_test_pred, axis=0)) / X_test.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))
"""
Based on the small discrepancy between training and test accuracy, we can conclude that the model 
only slightly overfits the training data. To further fine-tune the model, we could change the number 
of hidden units, values of the regularization parameters, learning rate, values of the decrease constant, 
or the adaptive learning
"""


"""
Now, let's take a look at some of the images that our MLP struggles with:
"""
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab= y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()    
"""
We should now see a 55× subplot matrix where the first number in the subtitles indicates the plot index, 
the second number indicates the true class label (t), and the third number stands for the predicted class label (p).
"""
