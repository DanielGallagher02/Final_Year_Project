# Convolutional Neural Network (CNN) for Crop Classification
# This CNN model architecture is inspired by the VGG16 model described by Simonyan and Zisserman in their paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition" (https://arxiv.org/abs/1409.1556).
# The implementation is adapted to suit the specific requirements of the agricultural image dataset used in my project.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define the CNN model structure
# Each convolutional layer is followed by a max-pooling layer and batch normalization
# which are standard practices for modern CNN architectures. For more details about these practices, please look at:
# "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (https://arxiv.org/abs/1502.03167).

# Defining the CNN model structure with Convolutional and Pooling layers,
# followed by Flatten and Dense layers for classification.
def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),  # Define the input shape of the images
        Conv2D(32, (3, 3), activation='relu', padding='same'),  # First Conv layer with 32 filters
        MaxPooling2D(2, 2),  # Pooling to reduce the spatial dimensions
        BatchNormalization(),  # Normalize the activations of the previous layer

        # Repeat the pattern of Conv -> Pooling -> BatchNorm with increasing filters
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Flatten(),  # Flatten the 3D output to 1D for the Dense layers
        Dense(512, activation='relu'),  # Dense layer with 512 neurons
        Dropout(0.5),  # Dropout for regularization to prevent overfitting
        Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
    ])

    # Compile the model with Adam optimizer and categorical_crossentropy loss for multi-class classification
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create data generators
def create_datagen(dir_path):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        dir_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'  # Use 'binary' if you have two classes
    )

# Main code block
if __name__ == "__main__":
    # Define input shape and number of classes
    input_shape = (224, 224, 3)
    num_classes = 5  # Update this to the actual number of classes you have
    
    # Assuming the base directory is where your 'processed' folder resides
    base_dir = '../data/processed/'  # Update this to your actual directory path

    # Creating data generators for each condition of Lettuce and type of Spinach
    # The paths should reflect the full directory structure for the training, validation,
    # and testing data for each plant or condition.

    # Grass:
    train_grass_gen = create_datagen(f'{base_dir}/Grass_processed/train')
    val_grass_gen = create_datagen(f'{base_dir}/Grass_processed/validation')
    test_grass_gen = create_datagen(f'{base_dir}/Grass_processed/test')

    # Lettuce with Potassium deficiency
    train_lettuce_k_gen = create_datagen(f'{base_dir}/Lettuce_processed/-K/train')
    val_lettuce_k_gen = create_datagen(f'{base_dir}/Lettuce_processed/-K/validation')
    test_lettuce_k_gen = create_datagen(f'{base_dir}/Lettuce_processed/-K/test')

    # Lettuce with Nitrogen Deficiency
    train_lettuce_n_gen = create_datagen(f'{base_dir}/Lettuce_processed/-N/train')
    val_lettuce_n_gen = create_datagen(f'{base_dir}/Lettuce_processed/-N/validation')
    test_lettuce_n_gen = create_datagen(f'{base_dir}/Lettuce_processed/-N/test')

    # Lettuce with Phosphorus Deficiency
    train_lettuce_p_gen = create_datagen(f'{base_dir}/Lettuce_processed/-P/train')
    val_lettuce_p_gen = create_datagen(f'{base_dir}/Lettuce_processed/-P/validation')
    test_lettuce_p_gen = create_datagen(f'{base_dir}/Lettuce_processed/-P/test')

    # Lettuce thats Fully Nutritional
    train_lettuce_fn_gen = create_datagen(f'{base_dir}/Lettuce_processed/FN/train')
    val_lettuce_fn_gen = create_datagen(f'{base_dir}/Lettuce_processed/FN/validation')
    test_lettuce_fn_gen = create_datagen(f'{base_dir}/Lettuce_processed/FN/test')

    # Data generators for Spinach types
    train_amaranth_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Amaranth Leaves/train')
    val_amaranth_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Amaranth Leaves/validation')
    test_amaranth_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Amaranth Leaves/test')

    # Black Nightshade of Spinach
    train_black_nightshade_gen = create_datagen(f'{base_dir}/Spinach_processed/Black Nightshade/train')
    val_black_nightshade_gen = create_datagen(f'{base_dir}/Spinach_processed/Black Nightshade/validation')
    test_black_nightshade_gen = create_datagen(f'{base_dir}/Spinach_processed/Black Nightshade/test')

    # Curry Leaves of Spinach
    train_curry_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Curry Leaves/train')
    val_curry_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Curry Leaves/validation')
    test_curry_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Curry Leaves/test')

    # Drumstick Leaves of Spinach 
    train_drumstick_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Drumstick Leaves/train')
    val_drumstick_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Drumstick Leaves/validation')
    test_drumstick_leaves_gen = create_datagen(f'{base_dir}/Spinach_processed/Drumstick Leaves/test')

    # Malabar Spinach of Spinach
    train_malabar_spinach_gen = create_datagen(f'{base_dir}/Spinach_processed/Malabar Spinach/train')
    val_malabar_spinach_gen = create_datagen(f'{base_dir}/Spinach_processed/Malabar Spinach/validation')
    test_malabar_spinach_gen = create_datagen(f'{base_dir}/Spinach_processed/Malabar Spinach/test')

    # Build and compile the CNN model using the function defined above
    model = build_model(input_shape, num_classes)

    # The training loop is based on standard TensorFlow Keras practices, as described in the TensorFlow documentation:
    # TensorFlow, "Train and evaluate with Keras", https://www.tensorflow.org/guide/keras/train_and_evaluate

    # This example uses the grass generator.
    # Training the model
    history = model.fit(
        train_grass_gen,
        validation_data=val_grass_gen,
        epochs=30  # Set the number of epochs for training
    )

    # Evaluate the model's performance on the test set
    evaluation_result = model.evaluate(test_grass_gen)
    print(f"Test Accuracy: {evaluation_result[1]*100:.2f}%")

    # Model saving is a standard feature in Keras, documentation for which can be found at:
    # TensorFlow, "Save and load models", https://www.tensorflow.org/tutorials/keras/save_and_load

    # Save the trained model for future use
    model.save("../models/crop_classification_model.h5")  # Update this to your actual model save path
