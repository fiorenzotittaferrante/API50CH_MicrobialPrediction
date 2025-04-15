import os, sys, random, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score

random.seed(123456)
warnings.filterwarnings("ignore")

os.makedirs("Result/Performance", exist_ok=True)
os.makedirs("Result/Images/Performance/", exist_ok=True)


def train_and_evaluate(X, y, params=None, param_grid=None, dataset_type='balanced'):
    """
    Trains a RandomForest model using Leave-One-Out (LOO) cross-validation for each carbohydrate. 
    Optionally performs hyperparameter tuning with GridSearchCV.

    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.DataFrame): Label dataset with binary values (0 or 1) for multiple labels.
        param_grid (dict, optional): Hyperparameter grid for RandomForest tuning. 
        params (pd.DataFrame, optional): Predefined hyperparameters for each label.
        dataset_type (str, optional): Miscellaneous information to include in the output. 
                              Defaults to an empty string.

    Returns:
        pd.DataFrame: A DataFrame containing performance metrics for each label.
        dict: A dictionary of trained models for each carbohydrate.
    """
    
    output_tmp = pd.DataFrame()
    trained_models = {}
    models = {"RF": [[], []]}  # Lista per i valori reali e predetti
    models_sel_feat = {"RF": [[], []]}  # Lista per i valori reali e predetti

    feature_importances = []    # Per feature selection. Ultima modifica

    output_columns = [
        "Carbohydrates", "Model", "Estimators", "Pos real", "Pos pred",
        "Precision", "Recall", "F1 score class 0", "F1 score class 1",
        "Micro", "Macro", "Balanced accuracy", "Dataset type"
    ]
    
    for carbohydrate in y.columns:
        print(f"Addestramento modello su carboidrato {carbohydrate}.")
        y_col = y[carbohydrate]
        loo = LeaveOneOut()

        # Se `params` è fornito e contiene il carboidrato, usa i parametri predefiniti
        if params is not None and carbohydrate in params.index:
            n_estimators = params.loc[carbohydrate, "Estimators"]
            rfclf = RandomForestClassifier(n_estimators=n_estimators, random_state=1234, bootstrap=False, n_jobs=-1)

        else:
            # Se `param_grid` è assente, assegna valori di default
            param_grid = param_grid or {"n_estimators": [32, 64, 128]}
            
            grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=loo, scoring="accuracy", n_jobs=-1)
            grid_search.fit(X, y_col)
            n_estimators = grid_search.best_params_["n_estimators"]
            rfclf = RandomForestClassifier(n_estimators=n_estimators, random_state=1234, bootstrap=False, n_jobs=-1)

        # Leave-One-Out Cross Validation
        for i, (train_index, test_index) in enumerate(loo.split(X)):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y_col.iloc[train_index], y_col.iloc[test_index]
            rfclf.fit(X_train, y_train)
            y_pred = rfclf.predict(X_test)

            models["RF"][0].append(y_test.iloc[0])  # Valore reale
            models["RF"][1].append(y_pred[0])       # Valore predetto

            ###### NEW
            # 2. Calcola le importanze basandoti solo sui dati di training
            # importances = rfclf.feature_importances_
            # threshold = np.percentile(importances, 35)  # Per esempio, seleziona le feature sopra il x° percentile
            # feature_importance_df = pd.DataFrame({
            #     'Feature': X.columns,
            #     'Importance': importances
            # }).sort_values(by='Importance', ascending=False)
            # selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature']
            
            # # 3. Crea un nuovo dataset con le sole feature selezionate
            # X_train_selected = X_train[selected_features]
            # X_test_selected = X_test[selected_features]

            # # 4. Riaddestra il modello sul training set con feature selezionate
            # rf_selected = RandomForestClassifier(n_estimators=n_estimators, random_state=1234, bootstrap=False, n_jobs=-1)
            # rf_selected.fit(X_train_selected, y_train)
            
            # # 5. Effettua la predizione sul test set
            # y_pred = rf_selected.predict(X_test_selected)

            # models_sel_feat["RF"][0].append(y_test.iloc[0])
            # models_sel_feat["RF"][1].append(y_pred[0])
            ######           

        new_row = calculate_metrics(
            y_true=models["RF"][0],
            y_pred=models["RF"][1],
            carbohydrate_name=carbohydrate,
            model_name="RF",
            n_estimators=n_estimators,
            dataset_type=dataset_type
        )

        #display(new_row)

        ##### new
        # new_row2 = calculate_metrics(
        #     y_true=models_sel_feat["RF"][0],
        #     y_pred=models_sel_feat["RF"][1],
        #     carbohydrate_name=carbohydrate,
        #     model_name="RF",
        #     n_estimators=n_estimators,
        #     dataset_type=dataset_type
        # )

        # display(new_row2)
        # models_sel_feat["RF"] = [[], []]
        #####

        models["RF"] = [[], []]
        output_tmp = pd.concat([output_tmp, new_row], ignore_index=True)
        #output_tmp = pd.concat([output_tmp, new_row2], ignore_index=True)   ########### Modificato in new_row2
        trained_models[carbohydrate] = rfclf

    output = pd.DataFrame(output_tmp, columns=output_columns)

    return output, trained_models


def train(dataframes, dataset_type):
    """
    Train function.

    Args:
        dataframes (dict): A dictionary where keys are carbohydrate names and values 
                           are DataFrames containing features and labels.
        dataset_type (str): A unique identifier for saving performance metrics.

    Returns:
        dict: A dictionary of trained models for each carbohydrate.
    """
    
    params = None
    param_grid = None
    params_path = f"./Result/Performance/performance_{dataset_type}.xlsx"
    trained_models = {}
    performance = pd.DataFrame(columns=[
        "Carbohydrates", "Model", "Estimators", "Pos real", "Pos pred",
        "Precision", "Recall", "F1 score class 0", "F1 score class 1",
        "Micro", "Macro", "Balanced accuracy", "Dataset type"
    ])
        
    if os.path.exists(params_path):
        print("Parametri migliori trovati.")
        params = pd.read_excel(params_path).set_index("Carbohydrates")
    else:
        param_grid = {"n_estimators": [16, 32, 64, 128]}     


    for carbohydrate, X in dataframes.items():
        y = X[[carbohydrate]]
        X = X.drop(columns=[carbohydrate])

        tmp, model = train_and_evaluate(X, y, params=params, param_grid=param_grid, dataset_type=dataset_type)

        trained_models.update(model)
        performance = pd.concat([performance, tmp], ignore_index=True)

    performance.set_index("Carbohydrates", inplace=True)

    with pd.ExcelWriter(params_path) as writer:
        performance.to_excel(writer, index=True)

    print(f"Performance salvate in {params_path}")

    return performance, trained_models


def calculate_metrics(y_true, y_pred, carbohydrate_name, model_name, n_estimators, dataset_type, n_sample=509):
    """
    Calculate various performance metrics for a classification model.

    Args:
        y_true (list or pd.Series): True labels of the test set.
        y_pred (list or pd.Series): Predicted labels from the model.
        carbohydrate_name (str): The name of the carbohydrate being predicted.
        model_name (str): The name of the model used for predictions.
        n_estimators (int): Number of estimators used in the model.
        dataset_type (str): Miscellaneous information for reporting.
        n_sample (int, optional): Total number of samples in the dataset (default is 509).

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metrics, including precision, 
                      recall, F1 scores, and balanced accuracy for the given model and carbohydrate.
    """

    if isinstance(y_true, list):
        y_true = pd.Series(y_true)

    if isinstance(y_pred, list):
        y_pred = pd.Series(y_pred)

    real_pos = (y_true == 1).sum(axis=0)
    real_neg = (y_true == 0).sum(axis=0)
    counts = y_pred.value_counts()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro")
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

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
        "Dataset type": dataset_type,
    }

    return pd.DataFrame([new_row])


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
    
    palette = sns.color_palette('colorblind', n_colors=len(dataframe['Dataset type'].unique()))
    category_color_map = {category: color for category, color in zip(dataframe['Dataset type'].unique(), palette)}
    
    legend_entries = {}
    
    for index, row in dataframe.iterrows():
        category = row['Dataset type']
        
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
        output_path = os.path.join(f"./Result/Images/Performance/{file_name}.png")
        plt.savefig(output_path)
    
    plt.show()
    plt.close()
