# transfer_model.py

# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Function to build the transfer learning model using VGG16 as the base model
def build_transfer_model(input_shape, num_classes):
    """
    Builds a transfer learning model using VGG16 pre-trained weights.
    The model is adapted for the task of crop classification based on agricultural images.
    References:
    - VGG16 Model Architecture: Simonyan & Zisserman (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556.
    - TensorFlow Documentation on Transfer Learning: https://www.tensorflow.org/tutorials/images/transfer_learning

    Parameters:
    - input_shape (tuple): The shape of input images.
    - num_classes (int): Number of classes in the target classification task.

    Returns:
    - A TensorFlow model instance compiled and ready for training.
    """
    # Load the VGG16 model pre-trained on ImageNet data, excluding the top fully connected layers
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # Reference for model weights: ImageNet Large Scale Visual Recognition Challenge.

    # Freeze the layers of the base model to prevent updating during the first phase of training
    for layer in base_model.layers:
        layer.trainable = False

    # Adding custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Adding dropout for regularization
    # Dropout reference: Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
    predictions = Dense(num_classes, activation='softmax')(x)

    # Creating the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compiling the model
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Compilation reference: Adam Optimizer and Categorical Crossentropy are standard practices in training neural networks.

    return model

# Main block to run if this script is executed as the main program
if __name__ == "__main__":
    # Assuming the base directory is where your 'processed' folder resides
    input_shape = (224, 224, 3)
    num_classes = 5  

    # Building the model
    model = build_transfer_model(input_shape, num_classes)

    print("Model built successfully with transfer learning components.")
    print("Due to challenges and resource constraints, this model integration is not fully operational.")
    print("Further research and development are required to fully realize transfer learning in this project.")
