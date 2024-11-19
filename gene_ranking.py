import os, sys, time, math
import pandas as pd
import numpy as np
import process_data
import sklearn, nltk

from collections import Counter
from sklearn.ensemble import RandomForestClassifier

# Import custom libraries.
sys.path.append('Code/')
from utility import save_to_excel, clean_ds_store
from utility import get_files_name, gene_coverage_importance
from utility import common_genes


def simple_train(X, y, params):
    """
    Trains a Random Forest classifier for each carbohydrate and calculates feature importance.

    Args:
        X (pd.DataFrame): The feature matrix containing input data.
        y (pd.DataFrame): A DataFrame containing target variables (carbohydrates) for training.
        params (pd.DataFrame): A DataFrame containing model parameters, including the number of estimators for each carbohydrate.

    Returns:
        dict: A dictionary where keys are carbohydrate names and values are arrays of feature importances corresponding to each carbohydrate.
    """
    
    feature_importance = {col: 0 for col in y.columns}
            
    for carbohydrate in y.columns:

        y_col = y[carbohydrate]
        n_estimators = params.loc[carbohydrate, 'Estimators']

        rfclf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=2, random_state=1234, bootstrap=False)
        rfclf.fit(X, y_col)
        feature_importance[carbohydrate] = rfclf.feature_importances_
        
    return feature_importance


def get_feature_importances(dataset_folder):
    """
    Calculate the importance of each feature for each carbohydrate and save the results to an Excel file.

    This function processes the dataset, trains models on both balanced and resampled datasets using the 
    `simple_train` function, and computes feature importances. The results are saved in a file located at 
    `Result/feature_importances.xlsx`, which contains the importance values for all features across all carbohydrates.

    Args:
        dataset_folder (str): The path to the folder containing the dataset to be processed.

    Returns:
        pd.DataFrame: A DataFrame containing feature importances for each carbohydrate.
    """
    
    X, y_balanced, resampled_dataframes, _ = process_data.main(dataset_folder)

    feature_importance = pd.DataFrame(columns=X.columns)
    feature_importance_balanced = pd.DataFrame(columns=X.columns)
    feature_importance_resampled = pd.DataFrame(columns=X.columns)


    # Normal training.
    if os.path.exists('./Result/Performance/performance_classic.xlsx'):
        performance_classic = pd.read_excel('./Result/Performance/performance_classic.xlsx').set_index('Carbohydrates')
        feature_importance_balanced = simple_train(X, y_balanced, performance_classic)

    else:
        print('Train firstly the models.')
        exit(-1)
        
        
    # Training with resampled dataset.
    if os.path.exists('./Result/Performance/performance_resampled.xlsx'):
        performance_resampled = pd.read_excel('./Result/Performance/performance_resampled.xlsx').set_index('Carbohydrates')

        for carbohydrate, df in resampled_dataframes.items():

            y_col = df[carbohydrate].to_frame()
            df = df.drop(carbohydrate, axis=1)

            tmp = simple_train(df, y_col, performance_resampled)
            tmp = pd.DataFrame(tmp).T
            tmp.columns = X.columns
            feature_importance_resampled = pd.concat([feature_importance_resampled, tmp], axis=0)

    else:
        print('Train firstly the models.')
        exit(-1)

    feature_importance_balanced = pd.DataFrame(feature_importance_balanced)
    feature_importance_resampled = pd.DataFrame(feature_importance_resampled)

    feature_importance = pd.concat([feature_importance_balanced, feature_importance_resampled], axis=0)
    save_to_excel(feature_importance, "./Result/feature_importances.xlsx", index=True)      

    return feature_importance


def process_strains_coverage(feature_importances):
    """
    Calculate the genome coverage and gene importance for each strain based on the provided feature importances.

    This function iterates through each carbohydrate in the `feature_importances` DataFrame, calculates the 
    gene coverage for each strain using the `gene_coverage_importance` function, and saves the results 
    in individual Excel files within the specified directory structure. Each strain's coverage information 
    is stored under its corresponding carbohydrate.

    Args:
        feature_importances (pd.DataFrame): A DataFrame containing feature importance values for different carbohydrates.

    Returns:
        None: The function does not return any value but saves the coverage results as Excel files in 
        the 'Result/Coverages/Gene_Importances/' directory.
    """

    coverages = None
    filemap = get_files_name()

    strains = filemap.keys()

    for carbohydrate, line in feature_importances.iterrows():
        
        start_time = time.time()
        
        folder = f'Result/Coverages/Gene_Importances/{carbohydrate}/'
        if not os.path.exists(f'Result/Coverages/Gene_Importances/{carbohydrate}/'):
            os.makedirs(folder)
        
        for strain in strains:

            genome_coverage = gene_coverage_importance(strain, filemap, feature_importances=line.to_frame().T)
            
            if isinstance(genome_coverage, str):
                # print(f'** Error: {genome_coverage}')
                pass
            else:
                genome_coverage = pd.DataFrame(genome_coverage).sort_values(by='Importance', ascending=False)
                genome_coverage.to_excel(os.path.join(folder, f'{strain}.xlsx'), index=False)

        end_time = time.time()
        print(f"\tExecution time for {carbohydrate}: {end_time - start_time} seconds.")


def analyze_gene_families():
    """
    Analyze gene families based on their importance across different strains for each carbohydrate.

    This function processes gene coverage data for each carbohydrate, calculates the positive and negative 
    importance of gene families, and computes a ratio to assess the relative significance of each family. 
    The results are saved in Excel files for each carbohydrate in the 'Result/Coverages/Common_Genes/' directory.

    Returns:
        None: The function does not return any value but saves the analysis results as Excel files containing 
        the gene family importances and ratios.
    """

    _, y = process_data.load_and_preprocess("Data/")

    output = pd.DataFrame(columns=['Gene family', 'Positive importance', 'Negative importance', 'Ratio'])
    path = 'Result/Coverages/Gene_Importances/'
    output_path = 'Result/Coverages/Common_Genes/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    y.columns = y.columns.str.strip() # Rimuove eventuali spazi vuoti

    for carbohydrate in os.listdir(path):
        
        print(f'\tCalculating common genes for carbohydrate {carbohydrate}:\t', end=' ')
        
        test_path = os.path.join(path, carbohydrate)
        rows = []
        
        positive_strain = y[y[carbohydrate] == 1].index.to_list()
        negative_strain = y[y[carbohydrate] == 0].index.to_list()
        positive_strain = [str(s) for s in positive_strain]
        negative_strain = [str(s) for s in negative_strain]
        positive_importances_mean = 0
        negative_importances_mean = 0
        
        genes = common_genes(test_path)
        print(f"foundend {len(genes)}.")
        
        # save_common_genes(genes, carbohydrate)
        
        for gene_family, dataframe in genes.items():
            dataframe['Strain'] = dataframe['Strain'].astype(str)

            positive_rows = dataframe[dataframe['Strain'].isin(positive_strain)]
            negative_rows = dataframe[dataframe['Strain'].isin(negative_strain)]
            ratio = None
            
            # Only gene families with an importance are considered.
            p = positive_rows['Importance'].mean()
            n = negative_rows['Importance'].mean()
            
            # if p is not None and n is not None:
                
            ratio = p/n
            A = p / math.sqrt(p**2 + n**2) # Calculation of the cosine between the point (p, n) and the origin
            B = 1 / math.sqrt(1**2 + 1**2) # Calculation of the cosine between the point (1, 1) and the origin
            diff_AB = B - A
                 
            row = {
                'Gene family': gene_family, 
                'Positive importance': p, 
                'Negative importance': n, 
                'Ratio': ratio,
                'A': A,
                'B': B,
                'B-A': diff_AB
            }
                                
            rows.append(row)

        output = pd.DataFrame(rows).sort_values(by=['Positive importance','Negative importance'], ascending=[False,False])

        save_to_excel(output, f'{os.path.join(output_path, carbohydrate)}.xlsx')
        
    print(f'\nCommon genes importances saved in {output_path}.')


def download_stop_words():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print("\n")


def main():
    """
    Main function to execute the gene ranking analysis workflow.

    This function orchestrates the entire process by performing the following tasks:
        1. Loads feature importances from an existing Excel file or calculates them if the file does not exist.
        2. Processes strain coverage based on the loaded or calculated feature importances.
        3. Analyzes gene families to evaluate their importances and ratios across different strains for each carbohydrate.

    Returns:
        None: This function does not return any value but executes the analysis and generates relevant output files.
    """

    feature_importances = None
    feature_importances_OD = None

    print("Calculate gene ranking...")

    directories = ["Data/BacDive/AFLP_perl", "Data/BacDive/gff", "Data/BacDive/ncbi", "Data/BacDive/Patric"]

    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Missing directory {directory}. Download firstly all files.")
            exit(-1)
    
    clean_ds_store()

    if os.path.exists('./Result/feature_importances.xlsx'):
        feature_importances = pd.read_excel('./Result/feature_importances.xlsx').set_index('Unnamed: 0').rename_axis(['Carbohydrates'])
    else:
        feature_importance = get_feature_importances("Data/")

    if os.path.exists('./Result/feature_importances_OD.xlsx'):
        feature_importances_OD = pd.read_excel('./Result/feature_importances_OD.xlsx').set_index('Unnamed: 0').rename_axis(['Carbohydrates'])

    print('Info: all result will be saved in Result/Coverages/')
    # process_strains_coverage(feature_importances)
    process_strains_coverage(feature_importances_OD)

    # download_stop_words()
    analyze_gene_families()


if __name__ == '__main__':
    main()
