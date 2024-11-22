import os, random, warnings, sys
import numpy as np
import pandas as pd
from itertools import product
from statistics import mode

import sklearn
from sklearn import tree
from sklearn.model_selection import LeaveOneOut, GridSearchCV

# Import custom libraries.
sys.path.append('Code/')
from utility import calculate_metrics, downsampling, save_to_excel


# used in train_wit_rf
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score


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


def train(X, y, negative_sensibility=0.55, positive_sensibility=0.75, k=50, best_params=None, excluded=''):
    """
    Train decision tree classifiers using Leave-One-Out (LOO) cross-validation with 
    outlier detection and feature selection. This function implements six feature selection 
    techniques to train individual decision trees, ultimately forming a forest of models and 
    selecting the best-performing one based on evaluation metrics.

    The function iterates over each carbohydrate in the target labels (`y`), excluding any 
    specified in the `excluded` parameter. For each carbohydrate, it trains multiple decision 
    trees based on positive samples and selected negative samples. After generating a forest 
    of decision trees for each feature selection method, the model evaluates each forest 
    and selects the one with the best performance.

    Args:
        X (pd.DataFrame): The feature dataset containing the input variables.
        y (pd.DataFrame): The target labels for training, where each column corresponds to 
                          a different carbohydrate.
        negative_sensibility (float, optional): Sensitivity threshold for selecting features 
                                                based on negative samples. Default is 0.55.
        positive_sensibility (float, optional): Sensitivity threshold for selecting features 
                                                based on positive samples. Default is 0.75.
        k (int, optional): The number of top features to select based on their impact ratios. 
                           Default is 50.
        best_params (pd.DataFrame, optional): A DataFrame containing optimized hyperparameters 
                                              for the models. If provided, it overrides the 
                                              default values for `negative_sensibility`, 
                                              `positive_sensibility`, and `k`.
        excluded (list, optional): A list of carbohydrates to exclude from training.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with performance metrics for each carbohydrate,
                            including precision, recall, F1 scores, and balanced accuracy.
            - dict: A dictionary containing the best-trained forest for each carbohydrate,
                    identified based on F1 scores.

    Function Details:
        - For each carbohydrate, Leave-One-Out cross-validation is performed on positive 
          samples, each time training on a subset and testing on a single sample.
        - Six different feature selection methods are used to train individual decision trees, 
          resulting in six distinct forests.
        - For each sample, a selected set of negative samples is added to account for potential 
          outliers.
        - Each forest is evaluated on F1 scores, and the forest with the highest F1 score 
          (for the positive class) is selected as the best-performing forest for that carbohydrate.
    """

    output = pd.DataFrame(columns=['Carbohydrates', 'Model', 'Estimators', 'Pos real', 'Pos pred', 'Precision', 'Recall', 'F1 score class 0', 'F1 score class 1', 'Micro', 'Macro', 'Balanced accuracy', 'Misc'])
    columns = ['Carbohydrates'] + list(X.columns)
    common_features = pd.DataFrame(columns=columns)
    trained_forests = {}

    for carbohydrate in y.columns:
        best_f1_score = 0
        best_forest_name = None
        best_forest = None
        best_tmp = None
        best_features = None

        if carbohydrate in excluded:
            continue
            
        if best_params is not None:
            negative_sensibility = best_params.loc[carbohydrate, 'negative_sensibility']
            positive_sensibility = best_params.loc[carbohydrate, 'positive_sensibility']
            k = best_params.loc[carbohydrate, 'k']
        
        y_col = y[carbohydrate]
        loo = LeaveOneOut() 
                    
        positive_X = X[y_col == 1]
        positive_y = y_col[y_col.index.isin(positive_X.index)]
        negative_X = X[y_col == 0]
        negative_y = y_col[y_col.index.isin(negative_X.index)]  
        
        np.random.seed(123456)

        # Collect trees for each feature selection method
        feature_forests = {f'FS{i}': [] for i in range(1, 7)}

        # Train trees on different features
        for i, (train_index, test_index) in enumerate(loo.split(positive_X)):

            X_train, X_test = positive_X.iloc[train_index], positive_X.iloc[test_index]
            y_train, y_test = positive_y.iloc[train_index], positive_y.iloc[test_index]
                       
            negative_indices = negative_X.index
            negative_index_to_add = np.random.choice(negative_indices)
            train_indices = negative_indices.drop(negative_index_to_add)
            test_indices = [negative_index_to_add]
            
            X_train_negative, y_train_negative = X.loc[train_indices], y[carbohydrate].loc[train_indices]
            X_test_negative, y_test_negative = X.loc[test_indices], y[carbohydrate].loc[test_indices]
            
            a = X_train.apply(lambda col: (col >= 1).sum()) / len(X_train)
            b = X_train_negative.apply(lambda col: (col == 0).sum()) / len(X_train_negative)
            
            # Feature selection for six methods
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
            
            # Training six trees with different feature sets
            for idx, (features, tree_model) in enumerate(
                zip(
                    [positive_selected_features_1, positive_selected_features_2, positive_selected_features_3, 
                     positive_selected_features_4, positive_selected_features_5, positive_selected_features_6],
                    [tree.DecisionTreeClassifier(max_depth=None, random_state=123456) for _ in range(6)]
                )
            ):
                selected_columns = X_train.columns[features == 1] if isinstance(features, pd.Series) else features
                if len(selected_columns) > 0:
                    tree_model.fit(X_train[selected_columns], y_train)
                    feature_forests[f'FS{idx + 1}'].append((tree_model, selected_columns))
        
        # Evaluate F1 score for each of the six forests
        for forest_name, forest in feature_forests.items():
            predictions = []
            
            for j in range(len(X)):
                X_test_sample = X.iloc[j]                
                # Use each tree's selected columns to predict, and combine predictions with majority vote
                votes = [model.predict(X_test_sample[selected_columns].to_frame().T)[0] for model, selected_columns in forest if set(selected_columns).issubset(X_test_sample.index)]
                predictions.append(mode(votes) if votes else 0)  # Majority vote, or 0 if no votes

            tmp = calculate_metrics(
                y[carbohydrate], predictions, 
                carbohydrate_name=carbohydrate, model_name='DT-FS', 
                n_estimators=1, 
                misc=f'Outlier Detection - {forest_name}', 
                n_sample=len(y[carbohydrate] == 1)
            )
            tmp['negative_sensibility'] = negative_sensibility
            tmp['positive_sensibility'] = positive_sensibility
            tmp['k'] = k

            f1 = tmp.loc[tmp["Carbohydrates"] == carbohydrate, "F1 score class 1"].iloc[0]
            if f1 > best_f1_score:
                best_f1_score = f1
                best_forest_name = forest_name
                best_forest = forest
                best_features = [selected_columns for _, selected_columns in feature_forests[forest_name]] # CHECK
                best_tmp = tmp

        # Update output dataframe with new statistics.
        if best_tmp is not None:
            output = pd.concat([output, best_tmp], ignore_index=True)

        trained_forests[carbohydrate] = best_forest

        # Intersect features extracted from the best function
        all_features = set(feature for features in best_features for feature in features)
        feature_matrix = pd.DataFrame(
            [
                {feature: 1 if feature in features else 0 for feature in all_features}
                for features in best_features
            ]
        )

        new_row = feature_intersection(feature_matrix, y=None, sensibility=1)
        if isinstance(new_row, pd.Series):
            new_row = {col: new_row.iloc[i] if i < len(new_row) else 0
                       for i, col in enumerate(common_features.columns[:-1])}

        new_row['Carbohydrates'] = carbohydrate
        common_features = pd.concat([common_features, pd.DataFrame([new_row])], ignore_index=True)
        print(common_features)

    return output, trained_forests, common_features.set_index('Carbohydrates')


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
    features = pd.DataFrame(columns=X.columns)

    all_trained_models = {}
    
    print(f"\t\tNumber of combinations: {len(param_combinations)}. Advancement -> ", end="")
    i = 1

    for params in param_combinations:
        neg_sensibility, pos_sensibility, k = params
    
        print(i, end=', ')
        i = i + 1

        performance, forest_models, features = train(X, y, negative_sensibility=neg_sensibility, positive_sensibility=pos_sensibility, k=k, excluded=excluded)        
        
        output = pd.concat([output, performance])

        for carbohydrate, trees in forest_models.items():
            if carbohydrate not in all_trained_models:
                all_trained_models[carbohydrate] = {}

            all_trained_models[carbohydrate][(neg_sensibility, pos_sensibility, k)] = {
                "trees": trees,
                "features": features.loc[carbohydrate]
            }


    best_models = {}
    best_features = {}
    for carbohydrate in output['Carbohydrates'].unique():
        best_params = output.loc[output['Carbohydrates'] == carbohydrate, ['negative_sensibility', 'positive_sensibility', 'k']].values[0]
        best_params_tuple = tuple(best_params)

        if best_params_tuple in all_trained_models[carbohydrate]:
            best_models[carbohydrate] = all_trained_models[carbohydrate][best_params_tuple]["trees"]
            best_features[carbohydrate] = all_trained_models[carbohydrate][(neg_sensibility, pos_sensibility, k)]["features"]

        else:
            print(f"\tNo model founded for the parameter {best_params_tuple} in {carbohydrate}.")

    output = output.sort_values(by=['Carbohydrates', 'F1 score class 1', 'F1 score class 0', 'Balanced accuracy', 'Misc'], ascending=[True, False, False, False, True], axis=0)
    output = output.iloc[range(0, len(output), len(param_combinations))]
    
    return output, best_models, best_features


def main(X, y):
    print("\tTraining outlier data.")

    # Output directory
    if not os.path.exists('./Result/Performance'):
        os.makedirs('./Result/Performance')


    ###########################################################################
    # Load and process data for regular Outlier

    performance_OD1 = None
    verbose_output1 = None
    trained_models_OD1 = {}
    best_features_OD1 = pd.Series()

    # params, searching = load_params('./Result/Performance/performance_OD1.xlsx')
    
    # if params is not None:
    #     print("\tBest parameters founded.")
    #     performance_OD1, trained_models_OD1, best_features_OD1 = train(X, y, best_params=params, excluded=['D-GLUcose', 'D-FRUctose', 'L-XYLose'])
    # else:
    #     param_grid = {
    #         'neg_sensibility': [0.5],
    #         'pos_sensibility': [0.65, 0.8],
    #         'k': [35] #, 60, 140],
    #     }
    #     performance_OD1, trained_models_OD1, best_features_OD1 = find_best_parameters(X, y, param_grid, excluded=['D-GLUcose', 'D-FRUctose', 'L-XYLose'])

    # save_to_excel(performance_OD1, './Result/Performance/performance_OD1.xlsx')
    # performance_OD1 = performance_OD1.reset_index()
    # performance_OD1 = process_performance(performance_OD1)

    # save_to_excel(best_features_OD1, './Result/feature_importances_OD1_backup.xlsx', index=True)


    ###########################################################################
    # Load and process data for high positivity rate Outlier

    performance_OD2 = None
    trained_models_OD2 = {}
    best_features_OD2 = pd.Series()

    if 'D-GLUcose' in y and 'D-FRUctose' in y:
        y_od2 = pd.concat([y['D-GLUcose'].to_frame(), y['D-FRUctose'].to_frame()], axis=1)

        od_resampled = downsampling(X, y_od2)

        params, searching = load_params('./Result/Performance/performance_OD2.xlsx')
        param_grid = {
            'neg_sensibility': [0.7], #, 0.8],
            'pos_sensibility': [0.65], #, 0.5],
            'k': [30, 50, 140],
        } if searching else None

        for carbohydrate, df in od_resampled.items():
            y_col = df[carbohydrate].to_frame()
            df = df.drop(carbohydrate, axis=1)

            if searching:
                tmp1, trained_models_OD2[carbohydrate], best_features_OD2[carbohydrate] = find_best_parameters(X=df, y=y_col, param_grid=param_grid)
            else:
                tmp1, trained_models_OD2[carbohydrate], best_features_OD2[carbohydrate] = train(df, y_col, best_params=params)
                tmp1 = process_performance(tmp1)

            performance_OD2 = pd.concat([performance_OD2, tmp1]) if performance_OD2 is not None else tmp1

        save_to_excel(performance_OD2, './Result/Performance/performance_OD2.xlsx')


    ###########################################################################
    # Load and process data for worst Outlier (only 2 positives over all dataset)

    performance_OD3 = None
    verbose_output3 = None
    trained_models_OD3 = {}
    best_features_OD3 = pd.Series()

    if 'L-XYLose' in y:

        param_grid = {
            'neg_sensibility': [0.7, 0.8],
            'pos_sensibility': [1],
            'k': [150],
        }

        y_od3 = y['L-XYLose'].to_frame()
        performance_OD3, trained_models_OD3, best_features_OD3 = find_best_parameters(X, y_od3, param_grid)

        save_to_excel(performance_OD3, './Result/Performance/performance_OD3.xlsx')


    ###########################################################################

    best_features_OD1 = pd.DataFrame(best_features_OD1).T
    best_features_OD2 = pd.DataFrame(best_features_OD2).T
    best_features_OD3 = pd.DataFrame(best_features_OD3).T

    trained_models = {**trained_models_OD1, **trained_models_OD2, **trained_models_OD3}
    performance_OD = pd.concat([performance_OD1, performance_OD2, performance_OD3], axis=0, ignore_index=False)
    best_features = pd.concat([best_features_OD1, best_features_OD2, best_features_OD3], axis=0, ignore_index=False)
    best_features = best_features.dropna(how='all')  # Remove eventually some empty rows

    save_to_excel(best_features, './Result/feature_importances_OD.xlsx', index=True) # Used in gene_ranking.py

    return trained_models

