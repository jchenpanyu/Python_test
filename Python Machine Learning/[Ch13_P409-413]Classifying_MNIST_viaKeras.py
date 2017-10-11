# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 13:43:40 2017
"Python Machine Learning"
Chapter 13
Page: 409-413
@author: vincchen
"""

"""
######    Classifying handwritten digits    ######
######    via Theano & Keras                ######
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
let's cast the MNIST image array into 32-bit format:
"""
import theano
theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

"""
Next, we need to convert the class labels (integers 0-9) into the one-hot format. 
Fortunately, Keras provides a convenient tool for this:
"""
from keras.utils import np_utils
print('First 3 labels: ', y_train[:3])
y_train_ohe = np_utils.to_categorical(y_train)
print('\nFirst 3 labels (one-hot):\n', y_train_ohe[:3])


"""
Now, we can get to the interesting part and implement a neural network. Here, we will use the 
same architecture as in Chapter 12, Training Artificial Neural Networks for Image Recognition. 
However, we will replace the logistic units in the hidden layer with hyperbolic tangent activation functions, 
replace the logistic function in the output layer with softmax, and add an additional hidden layer. 
Keras makes these tasks very simple, as you can see in the following code implementation:
"""
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

np.random.seed(1)
model = Sequential()
model.add(Dense(input_dim=X_train.shape[1], output_dim=50, init='uniform', activation='tanh'))
model.add(Dense(input_dim=50, output_dim=50, init='uniform', activation='tanh'))
model.add(Dense(input_dim=50, output_dim=y_train_ohe.shape[1], init='uniform', activation='softmax'))
sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
"""
First, we initialize a new model using the Sequential class to implement a feedforward neural network. 
Then, we can add as many layers to it as we like. However, since the first layer that we add is the input layer, 
we have to make sure that the input_dim attribute matches the number of features (columns) in the 
training set (here, 768). Also, we have to make sure that the number of output units (output_dim) and 
input units (input_dim) of two consecutive layers match. In the preceding example, we added two 
hidden layers with 50 hidden units plus 1 bias unit each. Note that bias units are initialized 
to 0 in fully connected networks in Keras. This is in contrast to the MLP implementation in Chapter 12, 
Training Artificial Neural Networks for Image Recognition, where we initialized the bias units to 1, 
which is a more common (not necessarily better) convention.

Finally, the number of units in the output layer should be equal to the number of unique class labels—the 
number of columns in the one-hot encoded class label array. Before we can compile our model, we also have to
define an optimizer. In the preceding example, we chose a stochastic gradient descent optimization, 
which we are already familiar with, from previous chapters. Furthermore, we can set values for the 
weight decay constant and momentum learning to adjust the learning rate at each epoch. Lastly, we set 
the cost (or loss) function to categorical_crossentropy. The (binary) cross-entropy is just the 
technical term for the cost function in logistic regression, and the categorical cross-entropy is 
its generalization for multi-class predictions via softmax.

After compiling the model, we can now train it by calling the fit method. Here, we are using 
mini-batch stochastic gradient with a batch size of 300 training samples per batch. We train the MLP 
over 50 epochs, and we can follow the optimization of the cost function during training by setting verbose=1.
The validation_split parameter is especially handy, since it will reserve 10 percent of the training data
(here, 6,000 samples) for validation after each epoch, so that we can check if the model is 
overfitting during training.
"""
model.fit(X_train, y_train_ohe, nb_epoch=50, batch_size=300, 
          verbose=1, validation_split=0.1, show_accuracy=True)

"""
To predict the class labels, we can then use the predict_classes method to return the class labels directly as integers:
"""
y_train_pred = model.predict_classes(X_train, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])

"""
Finally, let's print the model accuracy on training and test sets:
"""
train_acc = float(np.sum(y_train == y_train_pred, axis=0)) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))
y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = float(np.sum(y_test == y_test_pred, axis=0)) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))
