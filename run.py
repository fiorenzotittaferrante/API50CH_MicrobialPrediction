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

    X, y_balanced, resampled_dataframes, y_od = process_data.main(dataset_folder)

    print("\nTraining...")
    balanced_models, unbalanced_models = train.main(X, y_balanced, resampled_dataframes)
    outlier_models = outlierDetection.main(X, y_od)

    models = balanced_models | unbalanced_models | outlier_models

    return models


if __name__ == "__main__":

    models_path = "Data/models.pkl"
    dataset_folder = "Data/"

    if os.path.exists(models_path):
        models = load_models(models_path)
    else:
        models = train_models(dataset_folder)
        save_models(models, models_path)

    X = load_data("Data/new_input.xlsx")
    result = evaluate.evaluate(models, X)
    
    print(f"\nPredictions saved in Result/prediction.xlsx.\n")
    save_to_excel(result, path="Result/prediction.xlsx")

    print(result)

