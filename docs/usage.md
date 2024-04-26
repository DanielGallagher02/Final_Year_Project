# Final Usage Instructions

## System Usage Documentation

### Overview
- This guide provides the necessary steps to operate the agricultural outcome optimization system, an integral component of the "Optimising Agricultural Outcomes Through Computer Vision" project.

### Prerequisites
1. Python (version 3.x preferred) installed on your device.
2. Compatible with systems that meet the hardware requirements outlined in the dissertation, optimized for MacBook Air with the M1 chip.

### Installation and Setup
1. Clone the project repository or download the source code to your local environment.
2. Open your terminal and navigate to the project directory.
3. Install the required dependencies using `pip install -r requirements.txt`.

## Executing the System

### Data Preprocessing
- Execute data_preprocessing.py to process raw image data into a usable format, stored in data/processed.

## Model Training and Transfer Learning
- Define and train the CNN model with `cnn_model.py`.
- For adapting pre-trained models to your dataset, run `transfer_model.py`.

## System Initialization
- The primary entry point for the system is `main.py`. Running this script will automatically manage the workflow, encompassing data preprocessing, model training, and evaluation.

## Model Evaluation
- Test model performance using `test_models.py` in the tests directory.
- Explore data characteristics and perform initial model tests with `data_exploration.ipynb` and `model_prototyping.ipynb`.

## Additional Documentation
- Design choices and methodologies are detailed in docs/design.md.
For an overview of the project structure and implementation, see Project Overview.txt.

## Project Directory Structure
- data/: Processed datasets ready for analysis and raw unprocessed data.
- docs/: Design documentation and usage instructions.
- models/: CNN and transfer learning model definitions.
- notebooks/: Notebooks for data exploration and model prototyping.
- src/: Houses main.py (the main application script) and other scripts for data preprocessing, training, evaluation, and utilities.
- tests/: Unit tests for model scripts.
- README.md: Project summary and setup guide.
- requirements.txt: Python package dependencies.

## Running the Main Application
- To operate the system, simply run `main.py`. This script is the central hub that integrates and streamlines the execution of all project components. It ensures the application functions as a cohesive whole, from initial data processing to the delivery of predictive insights.
