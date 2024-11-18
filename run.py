import os, sys
import process_data, train, outlierDetection, evaluate

# Import custom libraries.
sys.path.append('Code/')
from utility import load_models, save_models, load_data, save_to_excel


def train_models(dataset_folder):
    """
    Train machine learning models using the provided dataset.

    Args:
        dataset_folder (str): The path to the folder containing the dataset files.

    Returns:
        dict: A dictionary where keys are model types and values are the trained models,
              including balanced, unbalanced and outlier models.
    """

    balanced_models = {}
    unbalanced_models = {}
    outlier_models = {}

    X, y_balanced, resampled_dataframes, y_od = process_data.main(dataset_folder)

    print("\nTraining...")
    # balanced_models, unbalanced_models = train.main(X, y_balanced, resampled_dataframes)
    outlier_models = outlierDetection.main(X, y_od)

    models = balanced_models | unbalanced_models | outlier_models

    return models


def predict(models, path_file):
    """
    Make predictions using the provided trained models and input data.

    Args:
        models (dict): A dictionary containing the trained models.
        path_file (str): The path to the input file with data for prediction.

    Returns:
        None: The predictions are saved to an Excel file in the `Result` directory.
    """

    X = load_data(path_file)
    result = evaluate.evaluate(models, X)
    
    print(f"\nPredictions saved in Result/prediction.xlsx.\n")
    save_to_excel(result, path="Result/prediction.xlsx")

    print(result)


if __name__ == "__main__":

    models_path = "Data/models.pkl"
    dataset_folder = "Data/"

    if os.path.exists(models_path):
        models = load_models(models_path)
    else:
        models = train_models(dataset_folder)
        save_models(models, models_path)

    # predict(models, "Data/new_input.xlsx")

