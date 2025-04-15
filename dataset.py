import os, re
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import NeighbourhoodCleaningRule

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

os.makedirs("Result/Images/Dataset/", exist_ok=True)


def process_data(dataset_folder):
    
    if dataset_folder is None:
        raise ValueError("No dataset directory specified.")
    
    carbohydrates = pd.read_excel("Data/REF_API50CH.xlsx")
    name_mapping = dict(zip(carbohydrates["Test number"], carbohydrates["Active ingredients"]))
    
    files = os.listdir(dataset_folder)
    if "X.xlsx" not in files or "Y.xlsx" not in files:
        raise FileNotFoundError("Dataset not found. Execute the appropriate notebook.")
    
    X = pd.read_excel(os.path.join(dataset_folder, "X.xlsx")).set_index("strain").astype(int)
    X[X >= 1] = 1
    
    y = pd.read_excel(os.path.join(dataset_folder, "Y.xlsx"))
    y = y.rename(columns=name_mapping)  
    y = y.fillna(0)  
    y = y.set_index("strain")  
    y = y.drop(columns=["CONTROL"], axis=1, errors='ignore')  
    y = y.astype(int)  
    y.columns = y.columns.str.strip()  
    
    return X, y


def get_balancing(y, draw=False, threshold_rebalance=0.4, threshold_OD=0.2):
    
    balancing = pd.DataFrame(y[y == 1].sum() / len(y), columns=["Balance"])
    balancing["Positives"] = y[y == 1].sum()
    balancing["Negatives"] = y[y == 0].count()
    balancing = balancing.sort_values(by="Balance")
    
    if draw:
        draw_balancing(y, balancing, threshold_rebalance, threshold_OD, y_distance_legend=-0.04)
    
    return balancing


def divide_dataset(y, balancing, threshold_rebalance=0.4, threshold_OD=0.2):

    # Extract labels with a positivity rate too low or too high (OD)
    od_label = balancing.loc[(balancing["Balance"] < threshold_OD)].index # | (balancing["Balance"] > 1 - threshold_OD)].index
    y_od = y[od_label]

    balancing = balancing.drop(od_label, axis=0)
    y = y.drop(od_label, axis=1)

    # Extract labels that need to be resampled based on the positivity rate
    resampled_label = balancing.loc[(balancing["Balance"] < threshold_rebalance) | (balancing["Balance"] > 1 - threshold_rebalance)].index
    y_resampled = y[resampled_label]
    y = y.drop(resampled_label, axis=1)

    return y, y_od, y_resampled
    

def downsampling(X, y, threshold_rebalance, threshold_OD, draw=False):
    warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.", category=pd.errors.PerformanceWarning)

    # Some statistics to compare resampled and not resampled dataset  
    stats = pd.DataFrame(y[y==1].sum() / len(y), columns=['Balance'])
    stats['Positives'] = y[y==1].sum()
    stats['Negatives'] = y[y==0].count()
    stats['Total'] = len(y)

    resampled_dataframes = {}

    for col in y.columns:

        ncr = NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=3, n_jobs=-1)
        X_resampled, y_resampled = ncr.fit_resample(X, y[col])

        # New dataframe with resampled dataset and the relative test's result (y)
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_df[col] = y_resampled
        resampled_dataframes[col] = resampled_df 
        
        stats.at[col, 'new Balance'] = y_resampled[y_resampled==1].sum() / len(y_resampled)
        stats.at[col, 'new Positives'] = y_resampled[y_resampled==1].sum()
        stats.at[col, 'new Negatives'] = y_resampled[y_resampled==0].count()
        stats.at[col, 'new Total'] = len(y_resampled)
    
    if draw:
        draw_balancing(y, stats, threshold_rebalance, threshold_OD, title='Number of positives/negatives for each carbohydrates (Normal)', label_P='Positives', label_N='Negatives', draw_threshold_lines=False, x_size=10, y_size=6, y_distance_legend=-0.1)
        draw_balancing(y, stats, threshold_rebalance, threshold_OD, title='Number of positives/negatives for each resampled carbohydrates (Resampled)', label_P='new Positives', label_N='new Negatives', draw_threshold_lines=False, x_size=10, y_size=6, y_distance_legend=-0.1)
    
    return resampled_dataframes


def draw_balancing(y, balancing, threshold_rebalance, threshold_OD, title='Class Balance', label_P='Positives', label_N='Negatives', draw_threshold_lines=True, x_size=10, y_size=16, y_distance_legend=0.04):
    
    title_fig = re.sub(r"\s*\([^)]*\)", "", title)

    fig, ax = plt.subplots(figsize=(x_size, y_size))
    
    bars = ax.barh(balancing.index, balancing[label_P], color='#00AA8D', label=label_P)
    bars2 = ax.barh(balancing.index, balancing[label_N], color='#D32F2F', label=label_N, left=balancing[label_P])
    
    if draw_threshold_lines:
        total = len(y) # Number of samples
        ax.axvline(x=total*threshold_OD, color='#0E4BEF', linestyle='--', label=f'Threshold for OD')
        # ax.axvline(x=total-total*threshold_OD, color='#0E4BEF', linestyle='--')
        ax.axvline(x=total*threshold_rebalance, color='purple', linestyle='--', label=f'Threshold for resampling')
        ax.axvline(x=total-total*threshold_rebalance, color='purple', linestyle='--')
    
    ax.set_xlabel('Number of samples', fontsize=13)
    ax.set_ylabel('Carbohydrates', fontsize=13)
    ax.set_title(title_fig, pad=20, fontsize=15) #, x=0.36)
    ax.set_yticks(balancing.index)
    ax.set_yticklabels(balancing.index, fontsize=11)
    
    for bar, pos, neg in zip(bars, balancing[label_P], balancing[label_N]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {pos:.0f}/{pos+neg:.0f}', ha='left', va='center', color='white', weight='bold')
    
    handles, labels = ax.get_legend_handles_labels()    
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, y_distance_legend), fontsize=12)
    plt.tight_layout()

    title = title.split()[-1].strip("()")
    output_path = os.path.join(f"./Result/Images/Dataset/{title}.png")
    plt.savefig(output_path)

    plt.show()


def generate_dataset(dataset_folder, draw=False, threshold_rebalance=0.4, threshold_OD=0.2):
    
    X, y = process_data(dataset_folder)
    balancing = get_balancing(y, draw, threshold_rebalance, threshold_OD)
    y_balanced, y_od, y_resampled = divide_dataset(y, balancing, threshold_rebalance, threshold_OD)
    resampled_dataframes = downsampling(X, y_resampled, threshold_rebalance, threshold_OD, draw)
    
    return {
        "balanced": {col: pd.concat([X, y_balanced[[col]]], axis=1) for col in y_balanced.columns},
        "semibalanced": resampled_dataframes,
        "unbalanced": {col: pd.concat([X, y_od[[col]]], axis=1) for col in y_od.columns},
    }


if __name__ == '__main__':
    dataset = generate_dataset("Data/", draw=True)
    print("Dataset generation completed.")
