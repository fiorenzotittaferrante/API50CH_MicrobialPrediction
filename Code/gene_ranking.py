import os, re, time, math
import pandas as pd
import numpy as np
import sklearn, nltk

from dataset import process_data, generate_dataset
from feature_selection import feature_intersection
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

# For function simplify_product_name()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

stop_words = set(stopwords.words('english'))



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
    
    params = params.set_index("Carbohydrates")
    carbohydrate = y.columns[0]

    if carbohydrate not in params.index:
        print(f"Carbohydrate '{carbohydrate}' non trovato in params. Skip.")
        return None

    y_col = y[carbohydrate]
    n_estimators = int(params["Estimators"].loc[[carbohydrate]].iloc[0])

    rfclf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=2, random_state=1234, bootstrap=False)
    rfclf.fit(X, y_col)
    feature_importances = rfclf.feature_importances_
    feature_importances = pd.DataFrame(feature_importances.reshape(1, -1), columns=X.columns)
    feature_importances['Carbohydrates'] = carbohydrate

    return feature_importances



def simple_train_feature_selection(X, y, params):
    """
    Trains a Random Forest classifier for each carbohydrate and calculates feature importance.

    Args:
        X (pd.DataFrame): The feature matrix containing input data.
        y (pd.DataFrame): A DataFrame containing target variables (carbohydrates) for training.
        params (pd.DataFrame): A DataFrame containing model parameters, including the number of estimators for each carbohydrate.

    Returns:
        dict: A dictionary where keys are carbohydrate names and values are arrays of feature importances corresponding to each carbohydrate.
    """

    params = params.set_index("Carbohydrates")
    carbohydrate = y.columns[0]

    if carbohydrate not in params.index:
        print(f"Carbohydrate '{carbohydrate}' non trovato in params. Skip.")
        return None

    positive_X = X.loc[X.index[y[carbohydrate] == 1]]
    negative_X = X.loc[X.index[y[carbohydrate] == 0]]

    positive_sensibility = params.loc[carbohydrate, 'positive_sensibility']
    negative_sensibility = params.loc[carbohydrate, 'negative_sensibility']
    fs_number = int(params.loc[carbohydrate, 'Dataset type'][-1])

    loo = LeaveOneOut()
    feature_importances_list = []

    for train_index, _ in loo.split(positive_X):
        X_train = positive_X.iloc[train_index]

        a = X_train.apply(lambda col: (col >= 1).sum()) / len(X)
        b = negative_X.apply(lambda col: (col == 0).sum()) / len(negative_X)

        if fs_number == 1:
            feature_importances = feature_intersection(X_train)
        elif fs_number == 2:
            feature_importances = feature_intersection(X_train, sensibility=positive_sensibility)
        elif fs_number == 3:
            negative_selected_features = feature_intersection(negative_X, sensibility=negative_sensibility, two_tail=False)
            positive_selected_features = feature_intersection(X_train, sensibility=positive_sensibility)
            feature_importances = (positive_selected_features - negative_selected_features).replace(-1, 0)
        elif fs_number == 4:
            feature_importances = feature_intersection(X_train, sensibility=positive_sensibility, two_tail=True)
        elif fs_number == 5:
            feature_importances = (a / b)
        else:
            feature_importances = (b / a)

        feature_importances_list.append(feature_importances.to_numpy())

    df = pd.DataFrame(feature_importances_list)
    mean_values = np.mean(df, axis=0)
    mean_df = pd.DataFrame(mean_values).T
    mean_df.columns = X.columns
    mean_df['Carbohydrates'] = carbohydrate

    return mean_df



def get_feature_importances(dataset, dataset_type, fs=False):
    """
    Calculate the importance of each feature for each carbohydrate and save the results to an Excel file.

    This function processes the dataset, trains models using the `simple_train` function, and computes 
    feature importances. The results are saved in a file located at `Result/feature_importances.xlsx`, 
    which contains the importance values for all features across all carbohydrates.
    
    Returns:
        pd.DataFrame: A DataFrame containing feature importances for each carbohydrate.
    """
    
    # Se fs è vero, viene estratta l'importanza delle feature usando la frequenza delle stesse
    if not fs:
        if os.path.exists(f'Result/Performance/performance.xlsx'):
            performance = pd.read_excel(f'Result/Performance/performance.xlsx')
            performance = performance[performance['Dataset type'] != 'unbalanced']
            pass
        else:
            raise Exception('Train firstly the models.')
    else:
        if os.path.exists(f'Result/Performance/performance_fs_{dataset_type}.xlsx'):
            performance = pd.read_excel(f'Result/Performance/performance_fs_{dataset_type}.xlsx')
    
    feature_importances = pd.DataFrame()
    for carbohydrate, X in dataset.items():
        y = X[[carbohydrate]]
        X = X.drop(columns=[carbohydrate])
        
        if not fs:
            tmp = simple_train(X, y, performance)
        else:
            tmp = simple_train_feature_selection(X, y, performance)

        feature_importances = pd.concat([feature_importances, tmp], ignore_index=True)

    with pd.ExcelWriter(f"./Result/feature_importances/{dataset_type}_importances.xlsx") as writer:
        feature_importances.to_excel(writer, index=False)

    return feature_importances



def get_files_name():
    """
    Retrieve the names of each .wri file in the specified directory and map them to a dictionary.

    Returns:
        dict: A dictionary where each key is the file's base name (without extension) and each value is the full .wri file name.
    """
    
    filemap = dict()
    files = [f for f in os.listdir('Data/BacDive/AFLP_perl') if os.path.isfile('Data/BacDive/AFLP_perl/'+f)]
    
    for f in files:
        if 'GCA' in f:
            filemap[f[:f.index('.')]] = f
        else:
            filemap[f.replace('.wri','')] = f
            
    return filemap



def find_file(file_name):
    """
    Search for the path of a .gff file by its name in specified folders.
    """
        
    folder_paths = ['Data/BacDive/gff/Patric/', 'Data/BacDive/gff/ncbi/Decompress/']

    for folder_path in folder_paths:
        for file in os.listdir(folder_path):
            if file.startswith(file_name):
                return os.path.join(folder_path, file)
   


def extract_genes(strain_path):
    """
    Extract gene information from a .wri file, returning a dataframe with details of each genome for the specified strain.

    Args:
        strain_path (str): The file path to the .wri file containing genome data for the strain.

    Returns:
        pd.DataFrame: A dataframe containing details of each gene, including columns for sequence, type, coordinates, helix, 
                      and gene-specific information (ID, product, and locus tag).
    """
    
    column_names = ['Sequence', 'Database', 'type', "5'", "3'", 'idk', 'helix', 'idk2', 'info']
    genome = pd.read_csv(strain_path, sep='\t', comment='#', header=None, names=column_names)
    genome = genome[genome['type'] == 'CDS']
    
    tmp = genome['info'].str.split(';', expand=True)        
    ids = tmp.apply(lambda x: next((val.split('=')[1] for val in x if 'ID=' in val), None), axis=1).to_frame()
    products = tmp.apply(lambda x: next((val.split('=')[1] for val in x if 'product=' in val), "N/A") if any('product=' in val for val in x if isinstance(val, str)) else "N/A", axis=1).to_frame()
    locus_tags = tmp.apply(lambda x: next((val.split('=')[1] for val in x if 'locus_tag=' in val), "N/A") if any('locus_tag=' in val for val in x if isinstance(val, str)) else "N/A", axis=1).to_frame()
    
    tmp = pd.concat([ids, products, locus_tags], axis=1)
    tmp.columns = ['ID', 'Product', 'Locus tag']

    genome.drop(columns='info', inplace=True)
    genome = pd.concat([genome, tmp], axis=1)

    return genome



def gene_coverage_importance(strain, filemap, aflp_min=50, aflp_max=500, feature_importances=None):
    """
    Calculate the DNA coverage for each strain, identify gene locations, 
    and evaluate whether genes are completely, partially, or not covered by AFLP. 
    Additionally, compute the importance of each gene based on the significance of the AFLP fragment in which the gene resides.

    Args:
        strain (str): The name of the strain to analyze.
        filemap (dict): A mapping of strain names to their corresponding .wri file names.
        aflp_min (int, optional): Minimum length of AFLP fragments to consider. Defaults to 50.
        aflp_max (int, optional): Maximum length of AFLP fragments to consider. Defaults to 500.
        feature_importances (pd.DataFrame, optional): A dataframe containing importance values for AFLP fragments.

    Returns:
        dict or str: A dictionary with gene coverage information for the strain, or an error message if the strain file is not found.
    """

    DNA_coverages = dict()
    DNA_importances = dict()
    features = []
    sequence = None
    
    genome_coverage = {
        'Sequence': [],
        'Sequence length': [],
        'Start': [],
        'End': [],
        'Helix': [],
        'Locus tag': [],
        'Product': [],
        'Type': [],
        'Captured nucleotides': [],
        'Coverage over genome length': [],
        'Importance': [],
    }
        
    # Open the strain.wri file and read the AFLP.
    # This files contains only information about Sequences, coordinates of each fragments and their lengths.
    if strain in filemap.keys():
        with open('Data/BacDive/AFLP_perl/' + filemap[strain], 'r') as file:            
            for line in file:
                row = line.strip().split('\t')

                if len(row) == 9:
                    frag_length = int(row[7])

                    if (frag_length >= aflp_min) and (frag_length <= aflp_max):
                        sequence = row[1]

                        if sequence not in DNA_coverages:
                            DNA_coverages[sequence] = [False for i in range(int(row[2]))]
                            DNA_importances[sequence] = [0 for i in range(int(row[2]))]
                        
                        start = int(row[4])
                        end = int(row[6])
                        start = min(start, end)
                        importance = feature_importances.loc[:, int(frag_length)].iloc[0]
                        max_importance = max(DNA_importances[sequence][start:start+frag_length])                        

                        if max_importance < importance:
                            for i in range(start, start+frag_length):
                                DNA_coverages[sequence][i] = True
                                DNA_importances[sequence][i] = importance
                        else:
                            for i in range(start, start+frag_length):
                                if DNA_importances[sequence][i] < importance:
                                    DNA_coverages[sequence][i] = True
                                    DNA_importances[sequence][i] = importance

    
    # Find the relative .gff file
    # This files gives information about where the genes are.
    strain_path = find_file(file_name=strain)

    if strain_path is None:
        return f'file {strain} not found.'
    
    genomes = extract_genes(strain_path)
        
        
    # sequence variable could be null due to .wri files which have sequence length less or greather then aflp_min and aflp_max.
    if sequence is not None:
        
        for index, row in genomes.iterrows():

            sequence_id = row.loc['Sequence']

            if 'accn|' in sequence_id:
                sequence_id = sequence_id[5::]

            if sequence_id not in DNA_coverages.keys():
                continue

            sequence_length = len(DNA_coverages[sequence_id])
            start = min(row.loc["5'"] - 1, row.loc["3'"] - 1)
            end = max(row.loc["5'"] - 1, row.loc["3'"] - 1)
            helix = row.loc['helix']            
            product = row.loc['Product']
            locus_tag = row.loc['Locus tag']            
            genome_length = end - start

            coverage = sum(DNA_coverages[sequence_id][start:end])
            coverage2 = coverage / genome_length
            genome_importance = np.mean(DNA_importances[sequence_id][start:end])

            if coverage2 == 0:
                classification = 'N'
            elif coverage2 < 1:
                classification = 'P'
            else:
                classification = 'T'                

            genome_coverage['Sequence'].append(sequence_id)
            genome_coverage['Sequence length'].append(sequence_length)
            genome_coverage['Start'].append(start+1)
            genome_coverage['End'].append(end+1)
            genome_coverage['Helix'].append(helix)
            genome_coverage['Locus tag'].append(locus_tag)
            genome_coverage['Product'].append(product)
            genome_coverage['Type'].append(classification)
            genome_coverage['Captured nucleotides'].append(f'{coverage}/{genome_length}')
            genome_coverage['Coverage over genome length'].append(coverage2)
            genome_coverage['Importance'].append(genome_importance)
                                
    return genome_coverage



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

    filemap = get_files_name()
    strains = filemap.keys()

    for carbohydrate, line in feature_importances.iterrows():
        start_time = time.time()
        
        folder = f'Result/Coverages/Gene_Importances/{carbohydrate}/'
        os.makedirs(folder, exist_ok=True)
        
        for strain in strains:
            genome_coverage = gene_coverage_importance(strain, filemap, feature_importances=line.to_frame().T)
            
            if isinstance(genome_coverage, str):
                # print(f'** Error: {genome_coverage}')
                pass
            else:
                genome_coverage = pd.DataFrame(genome_coverage).sort_values(by='Importance', ascending=False)
                with pd.ExcelWriter(os.path.join(folder, f'{strain}.xlsx')) as writer:
                    genome_coverage.to_excel(writer, index=False)

        end_time = time.time()
        print(f"\tExecution time for {carbohydrate}: {end_time - start_time} seconds.")



def simplify_product_name(product_name, common_words):
    """
    Preprocess the gene product name by removing special characters and common words.

    Args:
        product_name (str): The original gene product name to be processed.
        common_words (set): A set of common words to be excluded from the processed name.

    Returns:
        str: A simplified string containing the remaining unique words after filtering.
    """

    words = word_tokenize(re.sub(r'[^\w\s]', ' ', product_name.lower()))  # Rimuovi caratteri speciali e tokenizza
    
    word_freq = Counter(words)
    
    unique_words = list(word_freq.keys())
    
    filtered_words = [word for word in unique_words if not word.startswith('fig') and word not in common_words]

    return ' '.join(filtered_words)



def common_genes(path):
    """
    Analyzes Excel files in the specified directory and aggregates gene information based on their 'Product'.

    Args:
        path (str): The directory path containing Excel files to analyze.

    Returns:
        dict: A dictionary where keys are unique gene product names and values are DataFrames containing aggregated gene information,
              including strain names, locus tags, helix information, start and end positions, and importance scores.
    """
    
    genes = dict()
    common_words = stop_words.union({'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '2c'})

    if os.path.isdir(path): # Path is a specific carbohydrate
        
        for file in os.listdir(path):            
            if file.endswith('.xlsx'):
                strain_genes = pd.read_excel(os.path.join(path, file))
                
                for index, row in strain_genes.iterrows():
                    if row['Type'] != 'N':
                        if isinstance(row['Product'], str) and row['Product'] != 'N/A':
                            product = re.sub(r'\([^)]*\)', '', row['Product'])
                            product = re.sub(r'/', '-', product)
                            product = simplify_product_name(product, common_words)

                            data = [[file[:-5], row['Locus tag'], row['Helix'], row['Start'], row['End'], row['Importance']]]
                            data = pd.DataFrame(data, columns=['Strain', 'Locus tag', 'Helix', 'Start', 'End', 'Importance'])

                        if product in genes:
                            genes[product] = pd.concat([genes[product], data], ignore_index=True)
                        else:
                            genes[product] = data

    return genes



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

    _, y = process_data("Data/")

    output = pd.DataFrame(columns=['Gene family', 'Positive importance', 'Negative importance', 'Ratio'])
    path = 'Result/Coverages/Gene_Importances/'
    output_path = 'Result/Coverages/Common_Genes/'

    os.makedirs(output_path, exist_ok=True)
    # y.columns = y.columns.str.strip() # Rimuove eventuali spazi vuoti

    for carbohydrate in os.listdir(path):
        
        print(f'\tCalculating common genes for carbohydrate {carbohydrate}:\t', end=' ')
        
        test_path = os.path.join(path, carbohydrate)

        if not os.path.isdir(path):
            continue

        rows = []
        positive_strain = y[y[carbohydrate] == 1].index.to_list()
        negative_strain = y[y[carbohydrate] == 0].index.to_list()
        positive_strain = [str(s) for s in positive_strain]
        negative_strain = [str(s) for s in negative_strain]
        
        genes = common_genes(test_path)
        print(f"foundend {len(genes)}.")
                
        for gene_family, dataframe in genes.items():
            dataframe['Strain'] = dataframe['Strain'].astype(str)

            positive_rows = dataframe[dataframe['Strain'].isin(positive_strain)]
            negative_rows = dataframe[dataframe['Strain'].isin(negative_strain)]
            ratio = None
            
            # Only gene families with an importance are considered.
            p = positive_rows['Importance'].mean()
            n = negative_rows['Importance'].mean()
                            
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

        output = pd.DataFrame(rows).sort_values(by=['Positive importance','Negative importance'], ascending=[False,False]).reset_index(drop=True)
        output.index += 1
        output.index.name = 'Rank'

        with pd.ExcelWriter(f'{os.path.join(output_path, carbohydrate)}.xlsx') as writer:
            output.to_excel(writer, index=True)
        
    print(f'\nCommon genes importances saved in {output_path}.')


def download_stop_words():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print("\n")



def calculate_gene_families():
    """
    Main function to execute the gene ranking analysis workflow.

    This function orchestrates the entire process by performing the following tasks:
        1. Loads feature importances from an existing Excel file or calculates them if the file does not exist.
        2. Processes strain coverage based on the loaded or calculated feature importances.
        3. Analyzes gene families to evaluate their importances and ratios across different strains for each carbohydrate.

    Returns:
        None: This function does not return any value but executes the analysis and generates relevant output files.
    """

    os.makedirs("Result/feature_importances", exist_ok=True)
    directories = ["Data/BacDive/AFLP_perl", "Data/BacDive/gff", "Data/BacDive/ncbi", "Data/BacDive/Patric"]
    
    for directory in directories:
        if not os.path.isdir(directory):
            raise Exception(f"Missing directory {directory}. Download firstly all files.")

    # Generazione dataset
    dataset = generate_dataset("Data/")
    balanced_dataset = dataset['balanced']
    semibalanced_dataset = dataset['semibalanced']
    unbalanced_dataset = dataset['unbalanced']

    print("Calculate feature importances.")
    if not os.path.exists("Result/feature_importances/feature_importances.xlsx"):
        feature_importances_balanced = get_feature_importances(balanced_dataset, "balanced")
        feature_importances_semibalanced = get_feature_importances(semibalanced_dataset, "semibalanced")
        feature_importances_unbalanced = get_feature_importances(unbalanced_dataset, "unbalanced", fs=True)

        feature_importances = pd.concat([feature_importances_balanced, feature_importances_semibalanced, feature_importances_unbalanced], axis=0)    
        with pd.ExcelWriter(f"./Result/feature_importances/feature_importances.xlsx") as writer:
            feature_importances.to_excel(writer, index=False)

    feature_importances = pd.read_excel("Result/feature_importances/feature_importances.xlsx").set_index("Carbohydrates")

    print("Processes strain coverage based on the feature importances.")
    process_strains_coverage(feature_importances)

    print("Analyzes gene families to evaluate their importances.")
    analyze_gene_families()

    print('Result saved in Result/Coverages/')

    