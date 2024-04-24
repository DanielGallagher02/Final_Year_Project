# model_training.py is responsible for the training of your CNN model.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from data_preprocessing import create_datagen  # Assume this function exists in data_preprocessing.py

def train_model():
    # Define the model architecture
    model = Sequential([
        # Add Convolutional layers, MaxPooling, Flatten, Dense layers as per your dissertation
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Create data generators
    train_gen = create_datagen('path_to_train_data')
    val_gen = create_datagen('path_to_val_data')
    
    # Train the model
    history = model.fit(train_gen, validation_data=val_gen, epochs=30)
    
    return model

# Additional functions for setting up the data generators, 
# applying data augmentation, and any other preprocessing steps can be defined here as well.