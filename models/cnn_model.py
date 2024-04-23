import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model structure
def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

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
    base_dir = 'path_to_processed_data_directory'  # Update this to your actual directory path

    # Define paths to the training, validation, and testing directories
    train_dir = f'{base_dir}/train'
    val_dir = f'{base_dir}/validation'
    test_dir = f'{base_dir}/test'

    # Build and compile the model
    model = build_model(input_shape, num_classes)

    # Create data generators for training, validation, and testing
    train_generator = create_datagen(train_dir)
    validation_generator = create_datagen(val_dir)
    test_generator = create_datagen(test_dir)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=30  # Update this if necessary
    )

    # Evaluate the model on the test set
    evaluation_result = model.evaluate(test_generator)
    print(f"Test Accuracy: {evaluation_result[1]*100:.2f}%")

    # Save the trained model
    model.save("path_to_save_model/crop_classification_model.h5")  # Update this to your actual model save path
