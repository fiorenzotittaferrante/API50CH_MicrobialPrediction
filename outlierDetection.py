import os, random, warnings, sys
import numpy as np
import pandas as pd
from itertools import product

import sklearn
from sklearn import tree
from sklearn.model_selection import LeaveOneOut, GridSearchCV

# Import custom libraries.
sys.path.append('Code/')
from utility import calculate_metrics, downsampling


def load_params(path):
    """
    Load parameters from an Excel file if it exists.

    Args:
        path (str): The file path of the Excel file to load.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with parameters set with 'Carbohydrates' as the index if found.
            - bool: A flag indicating whether the parameters were found (True if not found, False if found).
    """
    if os.path.exists(path):
        return pd.read_excel(path).set_index('Carbohydrates'), False
    
    return None, True


def process_performance(df, group_size=6):
    """
    Process the performance DataFrame by sorting and selecting rows.

    Args:
        df (pd.DataFrame): The DataFrame containing performance metrics.
        group_size (int): The number of rows to skip after every selected row.

    Returns:
        pd.DataFrame: A DataFrame containing only the first row of each ordered group.
    """
    df = df.sort_values(by=['Carbohydrates', 'Precision', 'F1 score class 1', 'Balanced accuracy', 'Misc'], 
                         ascending=[True, False, False, False, True])
    indices = range(0, len(df), group_size)

    return df.iloc[indices]


def feature_intersection(X, y, sensibility=1, two_tail=False):
    """
    Calculates common features for the specified class based on their frequency in the dataset.

    This function identifies features in the dataset that occur frequently enough to be considered 
    impactful, based on a specified sensitivity threshold. It returns a DataFrame indicating 
    which features meet the criteria.

    Args:
        X (pd.DataFrame): Input dataset containing features.
        y (pd.Series): Labels associated with the input dataset (not used in current implementation).
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


def train(X, y, negative_sensibility=0.55, positive_sensibility=0.75, k=50, best_params=None, excluded=None):
    """
    Train decision tree classifiers using a Leave-One-Out (LOO) cross-validation approach 
    with outlier detection and feature selection based on specified sensitivity thresholds.

    This function iterates over each carbohydrate in the provided labels (`y`), excluding 
    any specified carbohydrates, and trains multiple decision tree models on the positive 
    samples while incorporating selected negative samples for each iteration. It computes 
    relevant metrics for model evaluation and returns a DataFrame containing performance metrics 
    for each carbohydrate.

    Args:
        X (pd.DataFrame): The feature dataset containing the input variables.
        y (pd.DataFrame): The target labels for the training, where each column corresponds 
                          to a different carbohydrate.
        negative_sensibility (float, optional): Sensitivity threshold for selecting negative 
                                                 features. Default is 0.55.
        positive_sensibility (float, optional): Sensitivity threshold for selecting positive 
                                                 features. Default is 0.75.
        k (int, optional): The number of top features to select based on their impact ratios. 
                           Default is 50.
        best_params (pd.DataFrame, optional): A DataFrame containing best hyperparameters for 
                                               the models. If provided, overrides the default 
                                               sensibility values and `k`.
        excluded (list, optional): A list of carbohydrates to exclude from training.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame containing performance metrics for each carbohydrate.
            - dict: A dictionary of trained models for each carbohydrate.
    """
        
    output = pd.DataFrame(columns=['Carbohydrates', 'Model', 'Estimators', 'Pos real', 'Pos pred', 'Precision', 'Recall', 'F1 score class 0', 'F1 score class 1', 'Micro', 'Macro', 'Balanced accuracy', 'Misc'])
    trained_models = {}

    for carbohydrate in y.columns:
        
        if excluded is None or carbohydrate in excluded:
            continue
            
        if best_params is not None:
            negative_sensibility = best_params.loc[carbohydrate, 'negative_sensibility']
            positive_sensibility = best_params.loc[carbohydrate, 'positive_sensibility']
            k = best_params.loc[carbohydrate, 'k']
        
        models = {
            'DT-FS': [[], [], [], [], [], [], []],
        }

        y_col = y[carbohydrate]
        loo = LeaveOneOut()     
                    
        positive_X = X[y_col == 1]
        positive_y = y_col[y_col.index.isin(positive_X.index)]
        
        negative_X = X[y_col == 0]
        negative_y = y_col[y_col.index.isin(negative_X.index)]  
        
        np.random.seed(123456)

        # Models for each feature selection technique.
        feature_models = {f'FS{i}': [] for i in range(1, 7)}

        tree_1 = tree.DecisionTreeClassifier(max_depth=None, random_state=123456)
        tree_2 = tree.DecisionTreeClassifier(max_depth=None, random_state=123456)
        tree_3 = tree.DecisionTreeClassifier(max_depth=None, random_state=123456)
        tree_4 = tree.DecisionTreeClassifier(max_depth=None, random_state=123456)
        tree_5 = tree.DecisionTreeClassifier(max_depth=None, random_state=123456)
        tree_6 = tree.DecisionTreeClassifier(max_depth=None, random_state=123456)
                  
        for i, (train_index, test_index) in enumerate(loo.split(positive_X)):

            X_train, X_test = positive_X.iloc[train_index], positive_X.iloc[test_index]
            y_train, y_test = positive_y.iloc[train_index], positive_y.iloc[test_index]
                       
            negative_indices = negative_X.index
            negative_index_to_add = np.random.choice(negative_indices)
            train_indices = negative_indices.drop(negative_index_to_add)
            test_indices = [negative_index_to_add]
            
            X_train_negative, y_train_negative = X.loc[train_indices], y[carbohydrate].loc[train_indices]
            X_test_negative, y_test_negative = X.loc[test_indices], y[carbohydrate].loc[test_indices]
            
            a = X_train.apply(lambda col: (col >= 1).sum())/len(X_train)
            b = X_train_negative.apply(lambda col: (col == 0).sum())/len(X_train_negative)
            
            # Features selection.
            negative_selected_features = feature_intersection(X_train_negative, y_train_negative, sensibility=negative_sensibility, two_tail=False)
            positive_selected_features_1 = feature_intersection(X_train, y_train)
            positive_selected_features_2 = feature_intersection(X_train, y_train, sensibility=positive_sensibility)
            positive_selected_features_3 = (positive_selected_features_2 - negative_selected_features).replace(-1, 0)
            positive_selected_features_4 = feature_intersection(positive_X, y_col, sensibility=positive_sensibility, two_tail=True)
            positive_selected_features_5 = (a/b).sort_values(ascending=False).index[:k]
            positive_selected_features_6 = (b/a).sort_values(ascending=False).index[:k]
                    
            X_train = pd.concat([X_train, X_train_negative], axis=0)
            y_train = pd.concat([y_train, y_train_negative], axis=0)           
            X_test = pd.concat([X_test, X_test_negative], axis=0)
            y_test = pd.concat([y_test, y_test_negative], axis=0)
            
            # Training and prediction.
            selected_columns_1 = X_train.columns[positive_selected_features_1 == 1]
            selected_columns_2 = X_train.columns[positive_selected_features_2 == 1]
            selected_columns_3 = X_train.columns[positive_selected_features_3 == 1]
            selected_columns_4 = X_train.columns[positive_selected_features_4 == 1]
            
            models['DT-FS'][0].extend(y_test)
            
            for tree_idx, (tree_model, selected_columns) in enumerate(
                [
                    (tree_1, selected_columns_1),
                    (tree_2, selected_columns_2),
                    (tree_3, selected_columns_3),
                    (tree_4, selected_columns_4),
                    (tree_5, positive_selected_features_5),
                    (tree_6, positive_selected_features_6),
                ]
            ):
                if len(selected_columns) == 0:
                    models['DT-FS'][tree_idx + 1].extend([0] * len(y_test))
                    continue

                tree_model.fit(X_train.loc[:, selected_columns], y_train)
                feature_models[f"FS{tree_idx + 1}"] = (tree_model, selected_columns)
                models['DT-FS'][tree_idx + 1].extend(tree_model.predict(X_test.loc[:, selected_columns]))

        trained_models[carbohydrate] = feature_models

        for i in range(1,7):
            tmp = calculate_metrics(models['DT-FS'][0], models['DT-FS'][i], carbohydrate_name=carbohydrate, model_name='DT-FS', n_estimators=1, misc=f'Outlier Detection - FS{i}', n_sample=len(y[carbohydrate] == 1))
            tmp['negative_sensibility'] = negative_sensibility
            tmp['positive_sensibility'] = positive_sensibility
            tmp['k'] = k
            output = pd.concat([output, tmp], ignore_index=True)
        
    return output, trained_models


def find_best_parameters(X, y, param_grid, excluded=''):
    """
    Finds the best parameters for training models on a given dataset by 
    evaluating combinations of specified parameter values. The function 
    performs training using the provided `train` function, captures performance 
    metrics for each parameter combination, and returns the best-performing 
    results.

    Args:
        X (pd.DataFrame): The feature dataset containing the input variables.
        y (pd.DataFrame): The target labels for the training, where each column 
                          corresponds to a different carbohydrate.
        param_grid (dict): A dictionary where keys are parameter names and 
                           values are lists of parameter values to be tested.
        excluded (list, optional): A list of carbohydrates to exclude from training.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with the best performing parameter combinations 
                            for each carbohydrate, based on selected metrics.
            - pd.DataFrame: A DataFrame with verbose output containing performance metrics 
                            for all parameter combinations evaluated during the search.
            - dict: A dictionary of trained models for each carbohydrate.
    """
    
    output = pd.DataFrame(columns=['Carbohydrates', 'Model', 'Estimators', 'Pos real', 'Pos pred', 'Precision', 'Recall', 'F1 score class 0', 'F1 score class 1', 'Micro', 'Macro', 'Balanced accuracy', 'Misc', 'negative_sensibility', 'positive_sensibility', 'k'])
    verbose_output = pd.DataFrame(columns=['Carbohydrates', 'Model', 'Estimators', 'Pos real', 'Pos pred', 'Precision', 'Recall', 'F1 score class 0', 'F1 score class 1', 'Micro', 'Macro', 'Balanced accuracy', 'Misc', 'negative_sensibility', 'positive_sensibility', 'k'])
    param_combinations = list(product(*param_grid.values()))

    all_trained_models = {}
    
    print(f"\tNumber of combinations: {len(param_combinations)} -> ", end="")
    i = 1

    for params in param_combinations:
        neg_sensibility, pos_sensibility, k = params
    
        print(i, end=', ')
        i = i + 1

        performance, trained_models = train(X, y, negative_sensibility=neg_sensibility, positive_sensibility=pos_sensibility, k=k, excluded=excluded)

        verbose_output = pd.concat([output, performance])
        
        performance = performance.sort_values(by=['Carbohydrates', 'F1 score class 1', 'F1 score class 0', 'Balanced accuracy', 'Misc'], ascending=[True, False, False, False, True], axis=0)
        indices = range(0, len(performance), len(performance['Misc'].unique()))
        performance = performance.iloc[indices]  
        output = pd.concat([output, performance])


        for carbohydrate, model_dict in trained_models.items():
            if carbohydrate not in all_trained_models:
                all_trained_models[carbohydrate] = {}
            all_trained_models[carbohydrate][(neg_sensibility, pos_sensibility, k)] = model_dict

    output_verbose = output.sort_values(by=['Carbohydrates', 'F1 score class 1', 'F1 score class 0', 'Balanced accuracy', 'Misc'], ascending=[True, False, False, False, True], axis=0)
    indices = range(0, len(output), len(param_combinations)) # Takes the first row of each ordered group.
    output = output.iloc[indices]

    best_models = {}
    for carbohydrate in output['Carbohydrates'].unique():
        best_technique = output.loc[output['Carbohydrates'] == carbohydrate, 'Misc'].values[0][-3:]
        best_params = output.loc[output['Carbohydrates'] == carbohydrate, ['negative_sensibility', 'positive_sensibility', 'k']].values[0]
        best_params_tuple = tuple(best_params)

        best_models[carbohydrate] = all_trained_models[carbohydrate][best_params_tuple][best_technique]

    # print(best_models)
    
    return output, output_verbose, best_models


def main(X, y):
    print("\tTraining outlier data.")

    # Output directory
    if not os.path.exists('./Result/Performance'):
        os.makedirs('./Result/Performance')


    ###########################################################################

    performance_OD1 = None
    verbose_output1 = None
    trained_models_OD1 = {}

    # Load and process data for general Outlier
    params, searching = load_params('./Result/Performance/performance_OD1.xlsx')
    
    if params is not None:
        print("\tBest parameters founded.")
        performance_OD1, trained_models_OD1 = train(X, y, best_params=params, 
                                                    excluded=['D-GLUcose', 'D-FRUctose', 'L-XYLose'])

        print(performance_OD1)
    else:
        param_grid = {
            'neg_sensibility': [0.5],
            'pos_sensibility': [0.65, 0.8],
            'k': [35, 60, 140],
        }
        performance_OD1, verbose_output, trained_models_OD1 = find_best_parameters(X, y, param_grid, excluded=['D-GLUcose', 'D-FRUctose', 'L-XYLose'])
        save_to_excel(verbose_output, './Result/Performance/performance_OD1_verbose.xlsx')

    save_to_excel(performance_OD1, './Result/Performance/performance_OD1.xlsx')
    performance_OD1 = performance_OD1.reset_index()
    performance_OD1 = process_performance(performance_OD1)


    ###########################################################################
    # Load and process data for high positivity rate Outlier
    performance_OD2 = None
    verbose_output2 = None
    trained_models_OD2 = {}

    if 'D-GLUcose' in y and 'D-FRUctose' in y:
        y_od2 = pd.concat([y['D-GLUcose'].to_frame(), y['D-FRUctose'].to_frame()], axis=1)
        od_resampled = downsampling(X, y_od2, threshold_rebalance=0.4, threshold_OD=0.2)

        params, searching = load_params('./Result/Performance/performance_OD2.xlsx')
        param_grid = {
            'neg_sensibility': [0.7, 0.8],
            'pos_sensibility': [0.5, 0.65],
            'k': [30, 50, 140],
        } if searching else None

        for carbohydrate, df in od_resampled.items():
            y_col = df[carbohydrate].to_frame()
            df = df.drop(carbohydrate, axis=1)

            if searching:
                tmp1, tmp2, trained_models_OD2[carbohydrate] = find_best_parameters(X=df, y=y_col, param_grid=param_grid)
                verbose_output2 = pd.concat([verbose_output2, tmp2]) if verbose_output2 is not None else tmp2
                save_to_excel(verbose_output2, './Result/Performance/performance_OD2_verbose.xlsx')
            else:
                tmp1, trained_models_OD2[carbohydrate] = train(df, y_col, best_params=params)
                tmp1 = process_performance(tmp1)

            performance_OD2 = pd.concat([performance_OD2, tmp1]) if performance_OD2 is not None else tmp1

        save_to_excel(performance_OD2, './Result/Performance/performance_OD2.xlsx')


    ###########################################################################
    # Load and process data for worst Outlier (only 2 positives over all dataset)

    performance_OD3 = None
    verbose_output3 = None
    trained_models_OD3 = {}

    if 'L-XYLose' in y:

        param_grid = {
            'neg_sensibility': [0.7, 0.8],
            'pos_sensibility': [1],
            'k': [150],
        }

        y_od3 = y['L-XYLose'].to_frame()
        performance_OD3, verbose_output3, trained_models_OD3 = find_best_parameters(X, y_od3, param_grid)

        save_to_excel(performance_OD3, './Result/Performance/performance_OD3.xlsx')
        save_to_excel(verbose_output3, './Result/Performance/performance_OD3_verbose.xlsx')


    ###########################################################################
    # Extract best models
    trained_models = {**trained_models_OD1, **trained_models_OD2, **trained_models_OD3}
    performance_OD = pd.concat([performance_OD1, performance_OD2, performance_OD3], axis=0, ignore_index=False)

    # for carbohydrate in performance_OD['Carbohydrates'].unique():
    #     best_technique = performance_OD.loc[performance_OD['Carbohydrates'] == carbohydrate, 'Misc'].values[0][-3:]
    #     trained_models[carbohydrate] = trained_models[carbohydrate][best_technique]


    return trained_models

