# main.py

# It should be the entry point to your application.
# It will integrate all the components of the project.

# import model_training 
# import model_evaluation
import data_preprocessing
import data_splitting
# import utils
import os

if __name__ == "__main__":
    try:
        print("Starting data preprocessing...")
        datasets = [
            ('../data/raw/Lettuce more detailed_', '../data/processed/Lettuce_processed', '.png'),
            ('../data/raw/Spinach more detailed_', '../data/processed/Spinach_processed', '.png'),
            ('../data/raw/Grass2_', '../data/processed/Grass_processed', '.jpg')
        ]
        for raw_dir, proc_dir, img_ext in datasets:
            # Check if processed directory exists and has files
            if os.path.exists(proc_dir) and len(os.listdir(proc_dir)) > 0:
                print(f"Processed data already exists in {proc_dir}, skipping preprocessing.")
            else:
                print(f"Processing dataset in {raw_dir}")
                data_preprocessing.preprocess_dataset(raw_dir, proc_dir, img_ext)
        print("Data preprocessing completed successfully.")
        
    except Exception as e:
        print(f"Error during data preprocessing: {e}")

    try:
        # Data Splitting
        print("Starting data splitting...")
        data_splitting.split_data()
        print("Data splitting completed successfully.")
        
    except Exception as e:
        print(f"Error during data splitting: {e}")

    try:
        # Model Training
        print("Starting model training...")
        # model = model_training.train_model()
        print("Model training completed successfully.")
        
    except Exception as e:
        print(f"Error during model training: {e}")

    try:
        # Model Evaluation
        print("Starting model evaluation...")
        # model_evaluation.evaluate_model(model)
        print("Model evaluation completed successfully.")
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")

    # Additional steps for deployment can be included here.