# data_preprocessing.py

import os
from PIL import Image, ImageFile

# Configure Pillow to handle truncated images, which can occur if the image file is incomplete or corrupted.
ImageFile.LOAD_TRUNCATED_IMAGES = True

def preprocess_dataset(raw_data_dir, processed_data_dir, image_ext):
    """
    Preprocesses images in a given directory by resizing and saving them to a new directory.
    It also checks if directories already exist to avoid recreating them.

    Parameters:
    - raw_data_dir (str): Directory containing the raw images.
    - processed_data_dir (str): Directory where processed images will be saved.
    - image_ext (str): The file extension of images to process (e.g., '.png', '.jpg').
    """

    # Check if the processed data directory exists; if not, create it.
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
        print(f"Created directory {processed_data_dir}")

    # Traverse the directory tree rooted at raw_data_dir.
    for subdir, dirs, files in os.walk(raw_data_dir):
        for file in files:
            # Process only files with the specified image extension.
            if file.lower().endswith(image_ext):
                try:
                    # Construct the full path to the image file.
                    img_path = os.path.join(subdir, file)
                    with Image.open(img_path) as img:
                        # Resize the image to 224x224 pixels using high-quality downsampling filter.
                        processed_img = img.resize((224, 224), Image.LANCZOS)
                        
                        # Replace 'raw' in the path with 'processed' to maintain directory structure.
                        processed_subdir = subdir.replace('raw', 'processed')
                        # Check if the processed subdirectory exists; if not, create it.
                        if not os.path.exists(processed_subdir):
                            os.makedirs(processed_subdir)
                            print(f"Created subdirectory {processed_subdir}")
                        
                        # Construct the save path and save the processed image there.
                        save_path = os.path.join(processed_subdir, file)
                        processed_img.save(save_path)
                        print(f"Saved processed image to {save_path}")
                except Exception as e:
                    # Handle exceptions (e.g., file not found, corrupted file) and report the failure.
                    print(f"Failed to process {file}: {e}")

    print(f"Processing complete for {processed_data_dir}. Processed images are saved in", processed_data_dir)

# Define datasets and their respective directories and image extensions
datasets = [
    ('../data/raw/Lettuce more detailed_', '../data/processed/Lettuce_processed', '.png'),
    ('../data/raw/Spinach more detailed_', '../data/processed/Spinach_processed', '.png'),
    ('../data/raw/Grass2_', '../data/processed/Grass_processed', '.jpg')
]

# Process each dataset using the preprocess_dataset function.
for raw_dir, proc_dir, img_ext in datasets:
    preprocess_dataset(raw_dir, proc_dir, img_ext)
