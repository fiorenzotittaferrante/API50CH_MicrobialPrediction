import os, pickle, warnings, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from imblearn.under_sampling import NeighbourhoodCleaningRule
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score

# For function simplify_product_name()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

stop_words = set(stopwords.words('english'))


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


def save_to_excel(df, path, index=False):
    """
    Save a DataFrame to an Excel file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        path (str): The file path where the Excel file will be saved.
        index (bool): Whether to include the index in the Excel file.
    """

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=index)


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



###############################################################################
#                         Utility for gene_ranking.py                         #
###############################################################################

def find_file(file_name):
    """
    Search for the path of a .gff file by its name in specified folders.

    Args:
        file_name (str): The base name of the .gff file to search for.

    Returns:
        str or None: Returns the full path of the .gff file if found, or None if the file is not located in any specified folder.
    """
        
    folder_paths = ['Data/BacDive/gff/Patric/', 'Data/BacDive/gff/ncbi/Decompress/']

    for folder_path in folder_paths:
        for file in os.listdir(folder_path):
            if file.startswith(file_name):
                return os.path.join(folder_path, file)
    
    return None



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


# Previous name: get_genomes()
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


# Previous name: get_genome_coverage
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

    return ' '.join(filtered_words)                                       # Ritorna le parole rimanenti come una stringa


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
            # print(file, end='\t')
            
            if file.endswith('.xlsx'):
                strain_genes = pd.read_excel(os.path.join(path, file)) #.set_index('Sequence')

                for index, row in strain_genes.iterrows():
                    if row['Type'] != 'N':
                        if isinstance(row['Product'], str) and row['Product'] != 'N/A':
                            product = re.sub(r'\([^)]*\)', '', row['Product'])
                            product = re.sub(r'/', '-', product) # Alcuni product come 'spermidine/putrescine' potrebbero creare problemi durante il salvataggio.
                            product = simplify_product_name(product, common_words)

                            data = [[file[:-5], row['Locus tag'], row['Helix'], row['Start'], row['End'], row['Importance']]]
                            data = pd.DataFrame(data, columns=['Strain', 'Locus tag', 'Helix', 'Start', 'End', 'Importance'])

                        if product in genes:
                            genes[product] = pd.concat([genes[product], data], ignore_index=True)
                        else:
                            genes[product] = data

                        # print(product)
                        # print(pd.DataFrame(genes[product]))

    return genes

