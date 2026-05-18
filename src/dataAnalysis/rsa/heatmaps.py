import seaborn as sns
import matplotlib as plt
from typing import Tuple
from src import config
from src.fileManagement.csvUtils import *
from src.fileManagement.jsonUtils import getJsonInfo

def heatMap(csv_path, correlation_type, specific_value: None | Tuple[list[str], list[str]] = None, save_path=None, show=True, codification=False, extension='png', dpi=100):
    
    if specific_value == None:
        data = pd.read_csv(csv_path)
    else:
        data = findInCsv(csv_path, specific_value[0], specific_value[1])
        data = dictToDataFrame(data, param=correlation_type, codification=codification)
    
    print(data.shape)
    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(data, vmin=-0.5, vmax=1.0)

    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(pad=0.8)
    if save_path != None: plt.savefig(save_path, format=extension, dpi=dpi)
    if show: plt.show()

if __name__ == "__main__":

    datasets = config.datasets
        
    save_folder=f'./dataStorage/processedResults/model-model'
    csv_folder = getJsonInfo(config.json_info_path)["rsaData"]

    for (dataset, subset) in datasets:

        correlation = 'spearman'

        extension = 'eps'
        
        dt = dataset.replace('/', '-')
        
        print(f"\n--- {dt} ---")
        
        print(f"{dataset} - {correlation}")
        heatMap(f'{csv_folder}/{dt}Data.csv', correlation, specific_value=(['dataset'], [f'{dt}({subset})']), save_path=f'{save_folder}/{correlation}/{dt}.{extension}', show=False, codification=True, extension=extension)
        
        #print(f"{dataset} - spearman")
        #heatMap(f'{csv_folder}/{dt}Data.csv', 'spearman', specific_value=(['dataset'], [f'{dt}({subset})']), save_path=f'{save_folder}/spearman/{dt}.{extension}', show=False, codification=True, extension=extension)