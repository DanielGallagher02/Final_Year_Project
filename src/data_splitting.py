# data_spilitting.py

# This script splits an image datasets into training, 
# validation, and test sets and copies the images into 
# corresponding subfolders for machine learning model preparation.

import os
import shutil
from sklearn.model_selection import train_test_split

# Function to copy files into training, validation, and test directories based on the split ratio
def copy_files(files, destination_folder, split_ratio):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Split the files into training and a combination of validation and test files
    train_files, test_val_files = train_test_split(files, train_size=split_ratio[0], random_state=42)
    # Split the combination into separate validation and test files
    val_files, test_files = train_test_split(test_val_files, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=42)

    # Copy the files into their respective directories
    for folder_name, file_set in zip(('train', 'validation', 'test'), (train_files, val_files, test_files)):
        subfolder = os.path.join(destination_folder, folder_name)
        # Ensure the subfolder exists
        os.makedirs(subfolder, exist_ok=True)
        for file in file_set:
            # Copy each file to the appropriate subfolder
            shutil.copy(file, os.path.join(subfolder, os.path.basename(file)))

# Function to process a directory and split the images into training, validation, and test sets
def process_directory(directory, split_ratio):
    # Walk through the directory
    for subdir, dirs, files in os.walk(directory):
        # Only process directories that do not contain other subdirectories (leaf directories)
        if not dirs:
            # Collect all image files from the directory
            image_files = [os.path.join(subdir, file) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # If image files are found, process them
            if image_files:
                print(f"Processing {subdir}...")
                copy_files(image_files, subdir, split_ratio)

# Main function to execute the data splitting and copying
def main():
    # Define the processed data directory and the split ratios for the sets
    processed_data_dir = '../data/processed/'  # Replace with your actual directory path
    split_ratio = (0.7, 0.15, 0.15)  # Ratio for train, validation, and test sets

    # Execute the processing on the defined directory
    process_directory(processed_data_dir, split_ratio)
    print(f"Data splitting and copying complete. Check the '{processed_data_dir}' directory for the results.")

# Check if the script is run as the main program and not imported as a module
if __name__ == "__main__":
    main()
