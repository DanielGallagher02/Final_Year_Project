# Optimising Agricultural Outcomes Through Computer Vision: A Deep Dive into Transfer Learning for Crop Predictions

Welcome to my **Final Year Project** for my Bachelor's in Computing. This project focuses on using advanced computer vision and transfer learning techniques to predict crop yields, specifically aimed at supporting small-scale farms. The project utilizes deep learning models like **Convolutional Neural Networks (CNNs)** and **VGG16** to address challenges related to limited agricultural datasets.

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

The project was developed using:
* **Python** for implementing machine learning models and handling data.
* **TensorFlow** and **Keras** for model training and deployment.
* **DJI Mini SE Drone** for capturing high-resolution field images for analysis.

The main goal is to demonstrate how combining transfer learning with computer vision can address agricultural challenges, such as data scarcity, and offer practical solutions for farmers.

## Technologies Used

### Model Architecture
- **VGG16 Pre-Trained Model**: Utilized for its deep feature extraction capabilities.
- **CNNs**: Implemented with Keras and TensorFlow for image classification and prediction.
- **Transfer Learning**: Applied to adapt the VGG16 model for crop yield prediction with limited training data.

### Data Preprocessing
- **Image Resizing**: Standardized all input images to 224x224 pixels.
- **Data Augmentation**: Techniques like flipping, rotation, and zooming applied to increase dataset diversity.
- **Normalization**: Applied to ensure uniformity across image data.

## Dataset Structure

The dataset consists of high-resolution images of crops (lettuce, spinach, and grass) collected via a **DJI Mini SE Drone**. It includes:
- **Raw Images**: Captured directly from the field, containing various environmental factors (lighting, shadows).
- **Processed Images**: Standardized and preprocessed images ready for CNN training.

## Installation

### Dependencies
- **Python 3.9+**
- **TensorFlow 2.8+**
- **Keras**
- **Pillow** (for image processing)

### Installing

