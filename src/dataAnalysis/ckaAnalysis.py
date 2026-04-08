from src.plot import *
import json
from src.codifications import *
import pandas as pd

def ckaHeatmap(json_path, save_path=None, show=True):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    data_matrix = np.zeros((27, 27))
    np.fill_diagonal(data_matrix, 1)
    
    for key, value in json_data.items():
        cod1, cod2 = key.split(' ')
        
        n1, l1 = int(cod1[:-1]), cod1[-1]
        n2, l2 = int(cod2[:-1]), cod2[-1]
        
        i, j = codToInstance(n1, l1)[0], codToInstance(n2, l2)[0]
        
        val = value["CKA"][0][0]
        
        data_matrix[i][j] = val
        data_matrix[j][i] = val
        
        
    df = pd.DataFrame(data_matrix, columns=getCods(), index=getCods())
    
    plt.figure(figsize=(10, 8))
    
    ax = seaborn.heatmap(df, vmin=-0.5, vmax=1.0)

    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(pad=0.8)
    if save_path != None: plt.savefig(save_path, dpi=100)
    if show: plt.show()
        
        
    
    

if __name__ == "__main__":
    
    data_folder = "dataStorage/ckaData"
    
    datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)]
    
    for (dt, subset) in datasets[1:2]: #pois estou fazendo apenas o cifar10 por enquanto
        dt_dir = f"{dt.replace('/', '-')}({subset})"
        
        json_path = f"{data_folder}/{dt_dir}/results.json"
    
        ckaHeatmap(json_path, save_path=f"dataStorage/processedResults/cka/{dt_dir}.png")