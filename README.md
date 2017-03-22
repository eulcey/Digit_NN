Neural Network for MNIST Digits
===============================

A Neural Network in Python
--------------------------

3-Layer neural network for handwritten digit recognition
with the sigmoid function as activation function

Training and test sets
----------------------

From http://yann.lecun.com/exdb/mnist/index.html

unzip sets and put binary files into ./data_sets

Format of Neural Network training_set and training_labels
------------------------

training_set: 2D NumPy array with rows as flattened image pixel values between -1 and 1
training_labels: 1D NumPy array with labels between 0 and 9

Different options
-----------------

- set TRAIN_NEW for training network everytime in test_network() or load weights instead
- use original images with 28x28 size or trim by 4 pix on each side
  for 20x20 size
- change amount of hidden layer neurons; standard amount is 30
- change amount of training cycles; standard is 5

Used Libraries
--------------

- math
- NumPy
- Matplotlib
