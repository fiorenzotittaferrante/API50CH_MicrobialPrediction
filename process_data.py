import os, sys, warnings
import pandas as pd

sys.path.append('Code/')
from utility import downsampling

warnings.filterwarnings("ignore")


def load_and_preprocess(dataset_folder):
    """
    Load and preprocess dataset.

    Args:
        dataset_folder (String): Path of the dataset folder.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): Binary input data with strains as the index.
            - y (pd.DataFrame): Processed output data with renamed columns, 
                                strains as the index, and without the "CONTROL" column.
    """

    if dataset_folder is None:
        print("No dataset directory specified.")
        exit(-1)

    files = os.listdir(dataset_folder)

    if "X.xlsx" not in files and "Y.xlsx" not in files:
        print("Dataset not downloaded. Execute the notebook Code/BacDive.ipynb.")
        return -1

    X = pd.read_excel(dataset_folder + "X.xlsx")
    y = pd.read_excel(dataset_folder + "Y.xlsx")
    carbohydrates = pd.read_excel("Data/REF_API50CH.xlsx")

    X = X.set_index("strain")
    X = X.astype(int)
    X[X >= 1] = 1

    name_mapping = dict(
        zip(carbohydrates["Test number"], carbohydrates["Active ingredients"])
    )
    y = y.rename(columns=name_mapping)
    y = y.fillna(0)
    y = y.set_index("strain")
    y = y.drop(["CONTROL"], axis=1)
    y = y.astype(int)
    y.columns = y.columns.str.strip()

    return X, y


def get_balance(X, y):
    """
    Computes and returns a DataFrame with the balance of positive and negative 
    samples for each label in the dataset y.

    Args:
        X (pd.DataFrame): Feature dataset (not used in the computation but included for consistency).
        y (pd.DataFrame): Label dataset with binary values (0 or 1).

    Returns:
        pd.DataFrame: A DataFrame containing:
            - "Balance": Proportion of positive samples for each label.
            - "Positives": Total count of positive samples (1s).
            - "Negatives": Total count of negative samples (0s).
    """

    balancing = pd.DataFrame(y[y == 1].sum() / len(y), columns=["Balance"])
    balancing["Positives"] = y[y == 1].sum()
    balancing["Negatives"] = y[y == 0].count()
    balancing = balancing.sort_values(by="Balance")

    return balancing


def divide_dataset(y, balancing, threshold_rebalance, threshold_OD):
    """
    Divides the dataset into three subsets based on the balance of positive samples.

    Args:
        y (pd.DataFrame): Label dataset with binary values (0 or 1).
        balancing (pd.DataFrame): DataFrame containing the balance information for each label.
        threshold_rebalance (float): Threshold to identify labels that need resampling.
        threshold_OD (float): Threshold to identify labels to be treated separately (OD).

    Returns:
        tuple: A tuple containing:
            - y (pd.DataFrame): The remaining labels after removing OD and resampled labels.
            - y_od (pd.DataFrame): Labels with a very low or very high positivity rate.
            - y_resampled (pd.DataFrame): Labels with a slightly low or slightly high positivity rate.
    """

    # Extract labels with a positivity rate too low or too high (OD)
    od_label = balancing.loc[
        (balancing["Balance"] < threshold_OD)
        | (balancing["Balance"] > 1 - threshold_OD)
    ].index
    y_od = y[od_label]

    balancing = balancing.drop(od_label, axis=0)
    y = y.drop(od_label, axis=1)

    # Extract labels that need to be resampled based on the positivity rate
    resampled_label = balancing.loc[
        (balancing["Balance"] < threshold_rebalance)
        | (balancing["Balance"] > 1 - threshold_rebalance)
    ].index
    y_resampled = y[resampled_label]
    y = y.drop(resampled_label, axis=1)

    y_resampled.reset_index()

    return y, y_od, y_resampled


def main(dataset_folder):
    print("Process dataset...")

    threshold_rebalance = 0.4  # Under-sampling: 20%-40% or 60%-80% of majority class
    threshold_OD = 0.2         # Outlier Detection: lower then 20% or greater than 80% of minority class

    X, y = load_and_preprocess(dataset_folder)
    print("\tDataset loaded.")

    balancing = get_balance(X, y)
    y_balanced, y_od, y_resampled = divide_dataset(
        y.copy(), balancing, threshold_rebalance, threshold_OD
    )
    resampled_dataframes = downsampling(X, y_resampled)
    print("\tDataset divided and downsampled.")

    return X, y_balanced, resampled_dataframes, y_od
