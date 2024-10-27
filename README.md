# Microbial Activity Prediction Pipeline

This project provides a machine learning pipeline for predicting microbial activity based on API 50 CH test data. The pipeline includes modules for data preprocessing, model training, outlier detection, and model evaluation, designed to handle both balanced and unbalanced datasets. 

Trained models are saved for reuse, enabling efficient microbial activity prediction on new test samples.

## Project Structure

- **`run.py`**: The main script that manages the workflow. It checks for existing models, trains and saves new models if necessary, and evaluates them on new data.
- **`process_data.py`**: Contains functions for data preprocessing, including data loading, balancing, and downsampling.
- **`train.py`**: Handles model training on both balanced and unbalanced datasets, returning a collection of trained models.
- **`outlierDetection.py`**: Module for detecting outliers by extracting common features from samples of the minority class.
- **`evaluate.py`**: Evaluation module with functions to assess model performance and make predictions on input data.
- **`utility.py`**: Contains utility functions for loading and saving models and data.

## Requirements
To create and use a virtual environment to isolate project dependencies, run:
``$ python -m venv ml_aflp50``

Then, activate the environment using:
``$ source ml_aflp50/bin/activate``

To set up the environment and install the necessary dependencies, run:
``$ pip install -r requirements.txt``

To deactivate the environment, run:
``$ deactivate``

## Usage

1. **Prepare Dataset**: Place all necessary dataset files in the `Data/` directory, including `Data/new_input.xlsx` for predictions.
2. **Run the Pipeline**: ``$ python run.py``

The script will check if trained models are available in Data/models.pkl. If no models are found, it will train new models using the data available in the Data/ directory.

Once the models are loaded or trained, the script will perform predictions on the new data in Data/new_input.xlsx and display the results.

## Models

The pipeline saves and loads trained models in a dictionary format, which includes:

- **Balanced Models**: Models trained on a balanced version of the microbial activity dataset, which aims to reduce class imbalances for more stable predictions.
  
- **Unbalanced Models**: Models trained on the original, unbalanced microbial activity dataset, preserving the natural distribution of the data.

- **Outlier Detection Models**: These models identify bacteria with low carbohydrate positivity, revealing shared characteristics and enhancing the robustness of the predictive framework.
