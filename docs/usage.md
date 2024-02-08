# Draft Usage instructions

# System Usage Documentation

## Overview
This document outlines the usage instructions for the agricultural outcome optimization system designed as part of the Final Year Project Dissertation "Optimising Agricultural Outcomes Through Computer Vision."

## Setting Up the Environment

### Prerequisites
- Ensure that you have Python installed on your system (Python 3.x is recommended).
- The system is designed to run on a MacBook Air with the M1 chip, but it should be compatible with other systems that meet the hardware requirements specified in the dissertation.

### Installation
1. Clone the repository to your local machine or download the source code.
2. Navigate to the project directory in your terminal.
3. Run `python3 -m venv venv` to create a virtual environment.
4. Activate the virtual environment:
   - On macOS and Linux: `source venv/bin/activate`
   - On Windows: `.\venv\Scripts\activate`
5. Install the required dependencies: `pip install -r requirements.txt`

## Running the System

### Data Preprocessing
- Use `data_preprocessing.py` to process the raw image data. The script will store processed data in the `data/processed` directory.

### Model Training
- Run `cnn_model.py` to define and train the Convolutional Neural Network model.
- For transfer learning, execute `transfer_model.py` with your dataset to fine-tune the pre-trained models.

### System Operation
- Start the system by running `main.py`. This script integrates all components and runs the entire workflow from data preprocessing to model evaluation.

## Testing the Model
- To test the model's performance, use the `test_models.py` script in the `tests` directory.
- The `data_exploration.ipynb` and `model_prototyping.ipynb` Jupyter notebooks can be used for exploratory data analysis and initial model testing, respectively.

## Additional Resources
- For more detailed design choices and methodologies, refer to the `docs/design.md`.
- You should look at `Project Overview.txt` for a more comprehensive overview of the project structure and implementation details.

