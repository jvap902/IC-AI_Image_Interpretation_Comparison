from src import config
from src.fileManagement.csvUtils import *
from src.fileManagement.jsonUtils import getJsonInfo

def heatMap(csv_path, correlation_type, specific_value: None | Tuple[list[str], list[str]] = None, save_path=None, show=True, codification=False, extension='png', dpi=100):
    
    if specific_value == None:
        data = paramDataFrameFromCsv(csv_path, correlation_type)
    else:
        data = findInCsv(csv_path, specific_value[0], specific_value[1])
        data = dataFrameFromData(data, param=correlation_type, codification=codification)
    
    print(data.shape)
    plt.figure(figsize=(10, 8))
    
    ax = seaborn.heatmap(data, vmin=-0.5, vmax=1.0)

    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(pad=0.8)
    if save_path != None: plt.savefig(save_path, format=extension, dpi=dpi)
    if show: plt.show()

if __name__ == "__main__":

    datasets = config.datasets
        
    save_folder=f'./dataStorage/processedResults/model-model'
    csv_folder = getJsonInfo()["rsaData"]

    for (dataset, subset) in datasets:

        extension = 'eps'
        
        dt = dataset.replace('/', '-')
        
        print(f"\n--- {dt} ---")
        
        print(f"{dataset} - pearson")
        heatMap(f'{csv_folder}/{dt}Data.csv', 'pearson', specific_value=(['dataset'], [f'{dt}({subset})']), save_path=f'{save_folder}/pearson/{dt}.{extension}', show=False, codification=True, extension=extension)
        
        #print(f"{dataset} - spearman")
        #heatMap(f'{csv_folder}/{dt}Data.csv', 'spearman', specific_value=(['dataset'], [f'{dt}({subset})']), save_path=f'{save_folder}/spearman/{dt}.{extension}', show=False, codification=True, extension=extension)