# This file turns the directory into a Python package.
# No code is needed here if you're only using it for this purpose.

import keras
import tensorflow
import pip

print(keras.__version__)  # This will print the Keras Version
print(tensorflow.__version__) #This will print the TensorFlow Version
print(pip.__version__) # This will print the PIP Version

import tensorflow as tf

# Test TensorFlow
hello = tf.constant('Hello, TensorFlow!')
tf.print(hello)