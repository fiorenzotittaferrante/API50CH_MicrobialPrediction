import os, pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score


def save_models(models, path):
    """
    Save the dictionary of models to a file.

    Args:
        models (dict): A dictionary where each key corresponds to a model type.
        path (str): The file path where the serialized models will be saved.

    Returns:
        None: The function does not return any value but writes the models to the specified file path.
    """

    try:
        with open(path, 'wb') as file:
            pickle.dump(models, file)
        print(f"\n\tModels successfully trained and saved to {path}.")
    
    except Exception as e:
        print(f"Error saving models: {e}")


def load_models(path):
    """
    Load the dictionary of models from a file.

    Args:
        path (str): The file path from which the models will be loaded.

    Returns:
        dict: A dictionary containing the models that were saved.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"\nThe file {path} does not exist.")
    
    try:
        with open(path, 'rb') as file:
            models = pickle.load(file)
        print(f"\nModels successfully loaded from {path}")

        return models
    
    except Exception as e:
        print(f"Error loading models: {e}")
        return None


def load_data(new_input_file):
    """
    Load and preprocess new input.

    Args:
        new_input_file (String): Path of the input file.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): Binary input data with strains as the index.
    """

    if new_input_file is None:
        print("No file specified.")
        exit(-1)

    if not new_input_file.endswith('.xlsx'):
        print("Input should be in .xlsx extension.")
        return -1

    X = pd.read_excel(new_input_file)

    X = X.set_index("strain")
    X = X.astype(int)
    X[X >= 1] = 1

    return X


def save_to_excel(df, path):
    """
    Save a DataFrame to an Excel file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        path (str): The file path where the Excel file will be saved.
    """
    
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False)


def calculate_metrics(
    y_test, y_pred, carbohydrate_name, model_name, n_estimators, misc, n_sample=509
):
    """
    Calculate various performance metrics for a classification model.

    Args:
        y_test (list or pd.Series): True labels of the test set.
        y_pred (list or pd.Series): Predicted labels from the model.
        carbohydrate_name (str): The name of the carbohydrate being predicted.
        model_name (str): The name of the model used for predictions.
        n_estimators (int): Number of estimators used in the model.
        misc (str): Miscellaneous information for reporting.
        n_sample (int, optional): Total number of samples in the dataset (default is 509).

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metrics, including precision, 
                      recall, F1 scores, and balanced accuracy for the given model and carbohydrate.
    """

    if isinstance(y_test, list):
        y_test = pd.Series(y_test)

    if isinstance(y_pred, list):
        y_pred = pd.Series(y_pred)

    real_pos = (y_test == 1).sum(axis=0)
    real_neg = (y_test == 0).sum(axis=0)
    counts = y_pred.value_counts()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average=None)
    f1_micro = metrics.f1_score(y_test, y_pred, average="micro")
    f1_macro = metrics.f1_score(y_test, y_pred, average="macro")
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    new_row = {
        "Carbohydrates": carbohydrate_name,
        "Model": model_name,
        "Estimators": n_estimators,
        "Pos real": f"{real_pos}/{n_sample}",
        "Pos pred": f"{counts.get(1, 0)}/{n_sample}",
        "Precision": precision,
        "Recall": recall,
        "F1 score class 0": round(f1_score[0], 3),
        "F1 score class 1": round(f1_score[1], 3),
        "Micro": round(f1_micro, 3),
        "Macro": round(f1_macro, 3),
        "Balanced accuracy": balanced_accuracy,
        "Misc": misc,
    }

    return pd.DataFrame([new_row])


def downsampling(X, y):
    """
    Performs downsampling on the given dataset using the Neighbourhood Cleaning Rule (NCR). 
    Each label in `y` is treated independently, and a new DataFrame is created for 
    each resampled label.

    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Label dataset with binary values (0 or 1).

    Returns:
        dict: A dictionary where keys are column names from `y`, 
              and values are DataFrames containing the resampled features 
              and corresponding labels.
    """
    
    warnings.filterwarnings(
        "ignore",
        message="DataFrame is highly fragmented.",
        category=pd.errors.PerformanceWarning,
    )

    resampled_dataframes = {}

    for col in y.columns:

        ncr = NeighbourhoodCleaningRule(
            sampling_strategy="majority", n_neighbors=3, n_jobs=-1
        )
        X_resampled, y_resampled = ncr.fit_resample(X, y[col])

        # New dataframe with resampled dataset and the relative test's result (y)
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_df[col] = y_resampled
        resampled_dataframes[col] = resampled_df

    return resampled_dataframes


def plot_performance(dataframe, size=(12, 7), file_name='plot', title='Performance all over models', metric='Balanced accuracy'):
    """
    Generate and display a bar plot of model performance metrics.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing performance metrics for various models.
        size (tuple): The size of the plot as a tuple (width, height).
        file_name (str): The name of the file to save the plot (without extension). If None, the plot will not be saved.
        title (str): The title of the plot.
        metric (str): The metric to plot from the DataFrame, such as 'Balanced accuracy' or 'F1 Score'.

    Returns:
        None: The function does not return any value but displays and optionally saves the plot to a specified file path.
    """

    plt.figure(figsize=size)
    plt.title(title, fontsize=15, pad=20)
    
    palette = sns.color_palette('colorblind', n_colors=len(dataframe['Misc'].unique()))
    category_color_map = {category: color for category, color in zip(dataframe['Misc'].unique(), palette)}
    
    legend_entries = {}
    
    for index, row in dataframe.iterrows():
        category = row['Misc']
        
        if category not in legend_entries:
            legend_entries[category] = plt.bar(index, row[metric], color=category_color_map[category], label=category)
        else:
            plt.bar(index, row[metric], color=category_color_map[category])
            
    if metric.startswith('F1'):
        metric = metric[:-8]
    
    plt.xlabel('Carbohydrates', fontsize=13)
    plt.ylabel(metric, fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if file_name is not None:
        os.makedirs(f"./Result/Images/", exist_ok=True)
        output_path = os.path.join(f"./Result/Images/{file_name}.png")
        plt.savefig(output_path)
    
    plt.show()
    plt.close()
    

def clean_ds_store(directory='.'):
    """
    Remove all '.DS_Store' files from the specified directory and its subdirectories.

    Args:
        directory (str): The root directory to search for '.DS_Store' files. Defaults to the current directory.

    Returns:
        None: The function does not return any value but deletes all found '.DS_Store' files from the directory.
    """

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                os.remove(file_path)