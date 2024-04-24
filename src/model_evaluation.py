# model_evaluation.py handles the evaluation of the trained model.

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from data_preprocessing import create_datagen  # Assume this function exists

def evaluate_model(model_path, test_data_path):
    model = load_model(model_path)
    
    # Create data generator for test set
    test_gen = create_datagen(test_data_path)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_gen)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    
    # Predict the test set
    y_pred = model.predict(test_gen)
    y_true = test_gen.classes
    print(classification_report(y_true, y_pred.argmax(axis=1)))

# Call evaluate_model function with the correct model path and test data path.