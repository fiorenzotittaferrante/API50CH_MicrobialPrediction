import os, sys, random, warnings
import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV

# Import custom libraries.
sys.path.append('Code/')
from utility import calculate_metrics

random.seed(123456)
np.random.seed(123456)
warnings.filterwarnings("ignore")

models = {
    "RF": [[], []],
}


def train(X, y, param_grid=None, params=None, misc=""):
    """
    Trains a RandomForest model using Leave-One-Out (LOO) cross-validation for each 
    label in the dataset. Optionally performs hyperparameter tuning with GridSearchCV.

    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Label dataset with binary values (0 or 1) for multiple labels.
        param_grid (dict, optional): Hyperparameter grid for RandomForest tuning. 
                                     Defaults to {'n_estimators': [32, 64, 128]}.
        params (pd.DataFrame, optional): Predefined hyperparameters for each label.
        misc (str, optional): Miscellaneous information to include in the output. 
                              Defaults to an empty string.

    Returns:
        pd.DataFrame: A DataFrame containing performance metrics for each label.
        dict: A dictionary of trained models for each carbohydrate.
    """

    output = pd.DataFrame(
        columns=[
            "Carbohydrates",
            "Model",
            "Estimators",
            "Pos real",
            "Pos pred",
            "Precision",
            "Recall",
            "F1 score class 0",
            "F1 score class 1",
            "Micro",
            "Macro",
            "Balanced accuracy",
            "Misc",
        ]
    )
    output_tmp = []
    trained_models = {}

    print(y)
    print("\n\n")

    for carbohydrate in y.columns:

        y_col = y[carbohydrate]
        loo = LeaveOneOut()

        print(f"{carbohydrate}: {params}")

        if params is not None:
            # Se in params esiste una riga con il carboidrato carbohydrate:
            n_estimators = params.loc[carbohydrate, "Estimators"]

            rfclf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_split=2,
                random_state=1234,
                bootstrap=False,
            )

        else:
            if param_grid is None:
                param_grid = {"n_estimators": [32, 64, 128]}

            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(),
                param_grid=param_grid,
                cv=loo,
                scoring="accuracy",
            )
            grid_search.fit(X, y_col)
            best_params = grid_search.best_params_
            n_estimators = best_params["n_estimators"]
            rfclf = RandomForestClassifier(**best_params)


        for i, (train_index, test_index) in enumerate(loo.split(X)):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y_col.iloc[train_index], y_col.iloc[test_index]               

            rfclf.fit(X_train, y_train)
            y_pred = rfclf.predict(X_test)

            models["RF"][0].extend(y_test)
            models["RF"][1].extend(y_pred.flatten())

        new_row = calculate_metrics(
            models["RF"][0],
            models["RF"][1],
            carbohydrate_name=carbohydrate,
            model_name="RF",
            n_estimators=n_estimators,
            misc=misc,
        )

        models["RF"][0] = []
        models["RF"][1] = []

        output_tmp.append(new_row)

        trained_models[carbohydrate] = rfclf

    output = pd.concat(output_tmp, ignore_index=True)

    return output, trained_models


def balanced_train(X, y):
    """
    Train a model using balanced data and store performance metrics.

    This function checks if a performance metrics file exists. If it does, it reads the parameters 
    from the file to be used for training. If not, it sets a default parameter grid. The function 
    then trains the model on the provided features and labels, stores the performance metrics in 
    an Excel file, and returns the trained models.

    Args:
        X (pd.DataFrame): Features dataset for training.
        y (pd.DataFrame): Labels dataset for training.

    Returns:
        dict: A dictionary of trained models for each carbohydrate.
    """

    params = None
    param_grid = None

    if os.path.exists("./Result/Performance/performance_classic.xlsx"):
        params = pd.read_excel(
            "./Result/Performance/performance_classic.xlsx"
        ).set_index("Carbohydrates")
    else:
        param_grid = {"n_estimators": [32, 64, 128]}

    performance_classic, trained_models = train(
        X, y, param_grid=param_grid, params=params, misc="Normal"
    )
    performance_classic = performance_classic.set_index("Carbohydrates")

    with pd.ExcelWriter("./Result/Performance/performance_classic.xlsx") as writer:
        performance_classic.to_excel(writer, index=True)

    return trained_models


def unbalanced_train(resampled_dataframes):
    """
    Train models on unbalanced data and store performance metrics.

    This function checks if a performance metrics file exists for previously trained models. 
    If it does, it reads the parameters from the file; if not, it sets a default parameter grid. 
    The function then iterates through each carbohydrate in the resampled dataframes, training 
    a model on the features and labels, storing the performance metrics in an Excel file, 
    and returning the trained models.

    Args:
        resampled_dataframes (dict): A dictionary where keys are carbohydrate names and values 
                                      are DataFrames containing features and labels.

    Returns:
        dict: A dictionary of trained models for each carbohydrate, with the carbohydrate name 
              as the key and the trained model as the value.
    """
    
    trained_models = {}
    params = None
    param_grid = None
    performance_resampled = pd.DataFrame(
        columns=[
            "Carbohydrates",
            "Model",
            "Estimators",
            "Pos real",
            "Pos pred",
            "Precision",
            "Recall",
            "F1 score class 0",
            "F1 score class 1",
            "Micro",
            "Macro",
            "Balanced accuracy",
            "Misc",
        ]
    )

    if os.path.exists("./Result/Performance/performance_resampled.xlsx"):
        params = pd.read_excel(
            "./Result/Performance/performance_resampled.xlsx"
        ).set_index("Carbohydrates")
    else:
        param_grid = {"n_estimators": [16, 32, 64, 128]}

    for carbohydrate, df in resampled_dataframes.items():

        y_col = df[carbohydrate].to_frame()
        df = df.drop(carbohydrate, axis=1)

        tmp, model = train(
            X=df, y=y_col, param_grid=param_grid, params=params, misc="Resampled"
        )

        trained_models = trained_models | model

        performance_resampled = pd.concat(
            [performance_resampled, tmp], ignore_index=True
        )

    performance_resampled = performance_resampled.set_index("Carbohydrates")

    with pd.ExcelWriter("./Result/Performance/performance_resampled.xlsx") as writer:
        performance_resampled.to_excel(writer, index=True)

    return trained_models


def main(X, y, resampled_dataframes):
    # Output directory
    if not os.path.exists('./Result/Performance'):
        os.makedirs('./Result/Performance')

    print("\tTraining balanced data.")

    balanced_trained_models = balanced_train(X, y)

    print("\tTraining unbalanced data.")
    unbalanced_trained_models = unbalanced_train(resampled_dataframes)

    return balanced_trained_models, unbalanced_trained_models