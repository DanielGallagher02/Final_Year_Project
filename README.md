# Optimising Agricultural Outcomes Through Computer Vision: A Deep Dive into Transfer Learning for Crop Predictions

Welcome to my **Final Year Project** for my (Hons) Bachelor's in Computing. This project focuses on using advanced computer vision and transfer learning techniques to predict crop yields, specifically aimed at supporting small-scale farms. The project utilizes deep learning models like **Convolutional Neural Networks (CNNs)** and **VGG16** to address challenges related to limited agricultural datasets.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
   - [Model Architecture](#model-architecture)
   - [Data Preprocessing](#data-preprocessing)
3. [Dataset Structure](#dataset-structure)
4. [Installation](#installation)
   - [Dependencies](#dependencies)
   - [Installing](#installing)
   - [Executing Program](#executing-program)
5. [Usage](#usage)
6. [Challenges & Future Work](#challenges--future-work)
7. [Contributors](#contributors)
8. [License](#license)
9. [Help](#help)
10. [Acknowledgments](#acknowledgments)

## Project Overview

This project explores the use of **computer vision** and **transfer learning** for improving crop yield predictions, with a particular focus on small-scale farming operations. By analyzing high-resolution crop images, the project aims to help farmers make informed decisions to improve productivity and resource management.

The project addresses key challenges small-scale farmers face, such as limited datasets and climate variability, by developing a **Convolutional Neural Network (CNN)** model utilizing **VGG16**. This pre-trained model is fine-tuned to recognize crop features and predict yields, even with small and diverse datasets.

The project was developed using:
* **Python** for implementing machine learning models and handling data.
* **TensorFlow** and **Keras** for model training and deployment.
* **DJI Mini SE Drone** for capturing high-resolution field images for analysis.

## Technologies Used

### Model Architecture
- **VGG16 Pre-Trained Model**: Utilized for its deep feature extraction capabilities. The model is fine-tuned for agricultural applications, especially in cases with limited training data.
- **CNNs**: Convolutional Neural Networks, implemented with Keras and TensorFlow, were used for classifying and predicting crop yields from the images.
- **Transfer Learning**: Transfer learning was applied to adapt the VGG16 model for our specific task, helping improve accuracy with limited agricultural datasets.

### Data Preprocessing
- **Image Resizing**: All input images were standardized to 224x224 pixels to ensure consistency in the dataset.
- **Data Augmentation**: Techniques such as flipping, rotation, and zooming were used to increase dataset diversity and improve model robustness.
- **Normalization**: Pixel values were normalized to enhance the model's ability to learn meaningful patterns from the images.

## Dataset Structure

The dataset consists of high-resolution images of crops, particularly **lettuce**, **spinach**, and **grass**, collected via a **DJI Mini SE Drone**. The images reflect various environmental conditions and stages of crop growth, which were essential in training the CNN model.

Key components:
- **Raw Images**: Unprocessed images, directly captured from the field, include noise from environmental factors like lighting and shadows.
- **Processed Images**: Standardized and preprocessed images, resized, augmented, and normalized for training and evaluation.

Additionally, external datasets were incorporated:
- **Lettuce NPK Dataset**: Sourced from Kaggle, containing detailed data on lettuce growth under varying nutrient compositions.
- **Spinach Dataset**: Specializing in images of spinach crops.
- **Grass Dataset**: Used to teach the model to distinguish between crops and surrounding flora.

## Installation

### Dependencies
- **Python 3.9+**
- **TensorFlow 2.8+**
- **Keras**
- **Pillow** (for image processing)

### Installing

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/DanielGallagher02/Final_Year_Project.git
   ```

2. **Install Required Python Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Dataset**: Place your drone-captured images in the data/raw folder.

### Executing Program

1. **Run Preprocessing Script**: To resize and normalize the images:
   ```bash
   python preprocess_images.py
   ```

2. **Train the Model**: After preprocessing, train the CNN with:
   ```bash
   python train_model.py
   ```

3. **Evaluate the Model**: After training, the model will evaluate its performance and generate predictions for crop yield:
   ```bash
   python evaluate_model.py
   ```

## Usage

- Once the model is trained, you can use it to predict crop yields by running the prediction script:
  ```bash
  python predict_crop_yield.py --image_path /path/to/new/image.jpg
  ```

- The prediction script will output the crop yield prediction based on the new image.

## Challenges & Future Work

### Challenges
- **Data Scarcity**: The project relied on limited data, a common issue for small-scale farms. Transfer learning helped, but gathering larger datasets is necessary for better accuracy.
- **Environmental Variability**: High variability in crop conditions across different farms presented challenges for model generalization.

### Future Work
- **Enhanced Model Generalization**: Future iterations will focus on improving the model's ability to generalize across diverse crop types and environmental conditions.
- **Real-Time Deployment**: I aim to integrate real-time predictions using TensorFlow Lite, allowing farmers to monitor crop yields on-site using mobile devices.

## Contributors

- **Daniel Gallagher** - Lead Developer and Researcher
- **Dr. Kevin Meehan** - Mentor for my thesis document in the 1st semester
- **Vini Vijayan** - Project Supervisor and mentor for my thesis document in the 2nd semester

## License

- This project is licensed under the MIT License - see the **LICENSE.md** file for details.

## Help

If you encounter any issues during the setup or execution of the program, ensure:
   - **Python** and required packages are installed.
   - **Dataset** is correctly placed in the working directory.
   - **Dependencies** are installed via **requirements.txt**.

## Acknowledgments

Special thanks to:
   - **Atlantic Technological University** for supporting this research and DJI Mini SE Drone technology used to capture high-resolution field images.


   

