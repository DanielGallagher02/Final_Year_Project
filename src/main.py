# main.py should be the entry point to your application.
# It will integrate all the components of the project.

import model_training 
import model_evaluation
import data_preprocessing
import data_splitting
import utils

if __name__ == "__main__":
    # Data Preprocessing
    data_preprocessing.preprocess_dataset()

    # Data Splitting
    data_splitting.split_data()

    # Model Training
    model = model_training.train_model()

    # Model Evaluation
    model_evaluation.evaluate_model(model)

    # Additional steps for deployment can be included here.