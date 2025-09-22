import sklearn
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from model_training import calculate_metrics
from itertools import product



def feature_intersection(X, sensibility=1, two_tail=False):
    """
    Calculates common features for the specified class based on their frequency in the dataset.

    This function identifies features in the dataset that occur frequently enough to be considered 
    impactful, based on a specified sensitivity threshold. It returns a DataFrame indicating 
    which features meet the criteria.

    Args:
        X (pd.DataFrame): Input dataset containing features.
        sensibility (float, optional): A threshold between 0 and 1 for feature selection. 
                                       Features are considered impactful if their frequency 
                                       is greater than or equal to this value. Default is 1.
        two_tail (bool, optional): If True, includes features that are infrequent, with a frequency 
                                    less than or equal to (1 - sensibility). Default is False.

    Returns:
        pd.DataFrame: A DataFrame with the same columns as X, where features are marked 
                      as 1 (impactful) or 0 (not impactful) based on the defined criteria.
    """

    if not (0 <= sensibility <= 1):
        raise ValueError("The parameter must be between 0 and 1.")

    out = pd.DataFrame(columns=X.columns)

    # For each feature, it checks how frequent it is and is evaluated as impactful (1) if greater than sensitivity
    selected_features = X.apply(lambda col: (col >= 1).sum() / len(X) >= sensibility).astype(int) 
    
    if two_tail:
        selected_features_2 = X.apply(lambda col: (col >= 1).sum() / len(X) <= 1-sensibility).astype(int) 
        selected_features = selected_features | selected_features_2 # Union.

    return selected_features



def train_feature_selection(dataframes, estimators, dataset_type, positive_sensibility=0.75, negative_sensibility=0.55, k=50):
    """
    Performs feature selection and trains models using LOO on each target.

    Args:
        dataframes (dict): {target_name: DataFrame} with features and label for each target.
        estimators (int): Number of trees for Random Forest; 0 uses Decision Tree.
        dataset_type (str): Label for the dataset category (e.g., 'balanced').
        positive_sensibility (float, optional): Sensitivity threshold for selecting positive features. Default is 0.75.
        negative_sensibility (float, optional): Sensitivity threshold for selecting negative features. Default is 0.55.
        k (int, optional): Number of top features to select based on ratios. Default is 50.

    Returns:
        pd.DataFrame: Aggregated metrics for each carbohydrate and feature selection method.
    """

    output = pd.DataFrame()

    key = 'DT-FS'
    if estimators != 0:
        key = 'RF-FS'

    for carbohydrate, X in dataframes.items():
        print(f"Extracting features for {carbohydrate}.")

        y = X[[carbohydrate]]
        X = X.drop(columns=[carbohydrate])

        positive_X = X.loc[X.index[y[carbohydrate] == 1]]
        negative_X = X.loc[X.index[y[carbohydrate] == 0]]
        positive_y = y.loc[y.index[y[carbohydrate] == 1]]
        negative_y = y.loc[y.index[y[carbohydrate] == 0]]
        
        models = {'DT-FS': [[], [], [], [], [], [], []]}

        loo = LeaveOneOut()
        
        # Train models on different features
        for _, (train_index, test_index) in enumerate(loo.split(positive_X)):

            X_train, X_test = positive_X.iloc[train_index], positive_X.iloc[test_index]
            y_train, y_test = positive_y.iloc[train_index], positive_y.iloc[test_index]
                       
            negative_indices = negative_X.index
            #num_negatives_test = min(1, len(negative_indices))  # Seleziona fino a 5 negativi nel test
            negative_test_indices = np.random.choice(negative_indices, 1, replace=False)
            train_indices = negative_indices.drop(negative_test_indices)
            test_indices = list(negative_test_indices)
            
            X_train_negative, y_train_negative = negative_X.loc[train_indices], negative_y[carbohydrate].loc[train_indices]
            X_test_negative, y_test_negative = negative_X.loc[test_indices], negative_y[carbohydrate].loc[test_indices]
            
            a = X_train.apply(lambda col: (col >= 1).sum()) / len(X_train)
            b = X_train_negative.apply(lambda col: (col == 0).sum()) / len(X_train_negative)
            
            # Feature selection for six methods
            negative_selected_features = feature_intersection(X_train_negative, sensibility=negative_sensibility, two_tail=False)
            positive_selected_features_1 = feature_intersection(X_train)
            positive_selected_features_2 = feature_intersection(X_train, sensibility=positive_sensibility)
            positive_selected_features_3 = (positive_selected_features_2 - negative_selected_features).replace(-1, 0)
            positive_selected_features_4 = feature_intersection(positive_X, sensibility=positive_sensibility, two_tail=True)
            positive_selected_features_5 = (a/b).sort_values(ascending=False).index[:k]
            positive_selected_features_6 = (b/a).sort_values(ascending=False).index[:k]

            X_train = pd.concat([X_train, X_train_negative], axis=0)
            y_train = pd.concat([y_train, y_train_negative], axis=0)
            X_test = pd.concat([X_test, X_test_negative], axis=0)
            y_test = pd.concat([y_test, y_test_negative], axis=0)

            models['DT-FS'][0].extend(y_test.values.flatten().tolist()) # Lista di interi di etichette
            
            # Training six rf with different feature sets
            for idx, features in enumerate(
                [positive_selected_features_1, positive_selected_features_2, positive_selected_features_3, 
                positive_selected_features_4, positive_selected_features_5, positive_selected_features_6]
            ):
                selected_columns = X_train.columns[features == 1] if isinstance(features, pd.Series) else features
                if len(selected_columns) == 0:
                    models['DT-FS'][idx + 1].extend([0] * len(y_test))
                    continue
                else:
                    if estimators == 0:
                        model = tree.DecisionTreeClassifier(max_depth=None, random_state=123456)
                    else:
                        model = RandomForestClassifier(n_estimators=estimators, random_state=123456)
                    model.fit(X_train[selected_columns], y_train)
                    models['DT-FS'][idx + 1].extend(model.predict(X_test.loc[:, selected_columns]).tolist()) # Lista di interi di predizioni

        for i in range(1,7):
            tmp = calculate_metrics(
                models['DT-FS'][0], models['DT-FS'][i], 
                carbohydrate_name=carbohydrate, model_name=key, 
                n_estimators=estimators, 
                dataset_type=f'{dataset_type} - FS{i}', 
                n_sample=len(y[carbohydrate] == 1)
            )
            tmp['negative_sensibility'] = negative_sensibility
            tmp['positive_sensibility'] = positive_sensibility
            tmp['k'] = k
            output = pd.concat([output, tmp], ignore_index=True)

    output = output.loc[output.groupby("Carbohydrates")["F1 score class 1"].idxmax()]

    return output



def find_best_parameters(datasets, param_grid, dataset_type):
    """
    Searches for the best feature selection parameters by evaluating all combinations.

    Args:
        datasets (dict): {target_name: DataFrame} with features and labels for each target.
        param_grid (dict): Dictionary of parameters to try (positive_sensibility, negative_sensibility, k, estimators).
        dataset_type (str): Label for the dataset category (e.g., 'balanced').

    Returns:
        pd.DataFrame: Best performance metrics for each carbohydrate, saved to Excel.
    """

    param_combinations = list(product(*param_grid.values()))
    best_performance = pd.DataFrame()

    for params in param_combinations:
        pos_sensibility, neg_sensibility, k, estimators = params

        performance = train_feature_selection(datasets, estimators, dataset_type, pos_sensibility, neg_sensibility, k)

        best_performance = pd.concat([best_performance, performance], ignore_index=True)

    best_performance = best_performance.loc[best_performance.groupby("Carbohydrates")["F1 score class 1"].idxmax()]

    output_path = f"./Result/Performance/performance_fs_{dataset_type}.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        best_performance.to_excel(writer, index=False)

    print(f"Performance salvate in {output_path}")
    
    return best_performance
