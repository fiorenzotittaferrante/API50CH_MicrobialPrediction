import pandas as pd
from statistics import mode
from sklearn.base import BaseEstimator


def evaluate(trained_models, data, output=None):
    """
    Evaluate the trained models on new input data and return predictions.

    Args:
        trained_models (dict): A dictionary where each key is a carbohydrate type
                               and the value is the corresponding trained model (or a tuple of model and features).
        data (pd.DataFrame): A pandas DataFrame containing the input data for evaluation.
        output (str, optional): The file path to save the output DataFrame with predictions. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions for each carbohydrate type,
                      indexed by the original data index.
    """
    
    print("\nEvaluating new data...")
    predictions = {}

    for carbohydrate, model in trained_models.items():

        if isinstance(model, list):    
            model_predictions = []

            for tree, features in model:
                pred = tree.predict(data.loc[:, features])
                model_predictions.append(pred)
            
            combined_predictions = [mode(pred) for pred in zip(*model_predictions)]
            predictions[carbohydrate] = combined_predictions

        elif isinstance(model, BaseEstimator):
            predictions[carbohydrate] = model.predict(data)

    out = pd.DataFrame(predictions)
    out.index = data.index

    return out