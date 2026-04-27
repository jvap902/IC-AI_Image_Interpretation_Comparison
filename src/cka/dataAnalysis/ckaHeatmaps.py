from src.fileManagement.csvUtils import *
from src.fileManagement.jsonUtils import getJsonInfo
from ..ckaFileRead import dfFromCkaJson
from src.codifications import *

def ckaHeatmap(json_path, save_path=None, show=True):

    df = dfFromCkaJson(json_path)
    
    plt.figure(figsize=(10, 8))
    
    ax = seaborn.heatmap(df, vmin=-0.5, vmax=1.0)

    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(pad=0.8)
    if save_path != None: plt.savefig(save_path, dpi=100)
    if show: plt.show()


def verifyIntegrity(json_path):
    data = getJsonInfo(json_path)
    
    keys = set(data.keys())
    
    cods = getCods()
    
    c = True
    
    for i in range(27):
        for j in range(i+1, 27):
            if(f"{cods[i]} {cods[j]}" not in keys):
                c = False
                print(f"ERRO - {cods[i]} {cods[j]} não encontrado")
    
    return c
            

if __name__ == "__main__":
    
    data_folder = "dataStorage/ckaData"
    
    datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)]
    
    for (dt, subset) in datasets[1:2]: #pois estou fazendo apenas o cifar10 por enquanto
        dt_dir = dtNameSubset((dt, subset))
        
        json_path = f"{data_folder}/{dt_dir}/results.json"
        
        if verifyIntegrity(f"{data_folder}/{dt_dir}/results.json"):
    
            ckaHeatmap(json_path, save_path=f"dataStorage/processedResults/cka/{dt_dir}.png")
        
        else:
            raise