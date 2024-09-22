# Optimising Agricultural Outcomes Through Computer Vision: A Deep Dive into Transfer Learning for Crop Predictions

## Project Overview
This project focuses on using **computer vision** and **transfer learning** to optimize crop yield predictions for small-scale farms. The goal is to address the challenges posed by limited datasets and environmental variability in agriculture. By leveraging deep learning techniques, we aim to provide small-scale farmers with a more reliable tool to enhance decision-making and boost productivity.

## Key Technologies
- **Convolutional Neural Networks (CNNs)**: Implemented with Keras and TensorFlow for deep learning and image analysis.
- **Transfer Learning**: Based on the VGG16 model to handle limited agricultural datasets and improve prediction accuracy.
- **Python**: Utilized for data processing and model implementation with key libraries such as NumPy, Pandas, and Pillow.

## Objectives
1. **Data Collection**: Utilizing high-resolution crop imagery collected via drones to create a diverse dataset.
2. **Image Preprocessing**: Standardizing image inputs using resizing, normalization, and augmentation techniques.
3. **Model Development**: Training a CNN using pre-trained models (VGG16) to predict crop yields.
4. **Evaluation**: Testing the model’s performance with small-scale farm datasets and refining it to enhance accuracy.
5. **Deployment**: Integrating the model into a real-world agricultural environment for crop monitoring.

## Dataset and Preprocessing
The project uses multiple datasets including **lettuce**, **spinach**, and **grass** crop images. The data undergoes several preprocessing steps to ensure consistency:
- **Image resizing** to 224x224 pixels.
- **Normalization** to enhance model learning.
- **Data augmentation** techniques like rotation, zooming, and flipping to simulate diverse agricultural conditions.

## Model Architecture
The project utilizes **Transfer Learning** by adapting the **VGG16** architecture:
- **Convolutional layers** for feature extraction.
- **Dropout and Batch Normalization** for regularization and improving model robustness.
- **Adam optimizer** and **categorical cross-entropy loss** for efficient training.

## Challenges and Future Work
- **Limited Dataset**: Transfer learning helps overcome data scarcity, but gathering more diverse agricultural images will improve the model’s accuracy.
- **Adaptability**: Future iterations aim to improve the model’s generalizability across various crops and environmental conditions.
- **Real-world Implementation**: Exploring the deployment of the model in resource-constrained environments with limited computational resources.


## How to Run the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/DanielGallagher02/agricultural-crop-prediction.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the preprocessing script:
    ```bash
    python preprocess_images.py
    ```
4. Train the model:
    ```bash
    python train_model.py
    ```

## Conclusion
This project demonstrates the potential of **AI** and **computer vision** in transforming agricultural practices. By improving crop yield prediction accuracy, small-scale farmers can make more informed decisions, resulting in more sustainable farming practices.
