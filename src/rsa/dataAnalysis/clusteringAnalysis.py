from src.fileManagement.csvUtils import *
from ...codifications import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
import json
from ...fileManagement.jsonUtils import getJsonInfo
from ...fileManagement.defaultPaths import jsonInfoPath
import os

datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)] #apenas datasets utilizados no artigo

instances = getInstances()

def appendCoephData(json_path, data):
    
    if os.path.isfile(json_path):
        with open(json_path, "r+") as f:
            json_data = json.load(f)
            
        json_data.update(data)
    else:
        json_data = data
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4, sort_keys=True)

def distMatirxModelNames(df):
    #dist = d(i,j) = (1-Corr(i,j))/2
    dist_matrix = np.empty([27, 27])
    
    model_names = []

    for idx, (key, row) in enumerate(df.iterrows()):
        model_names.append(key)
        
        for i, instance in enumerate(instances):
            dist_matrix[idx][i] = (1 - row[instance])/2
    
    model_names = list(dict.fromkeys(model_names))
    
    model_names = [getModelTrainStr(model[0], model[1], model[2]) for model in model_names]
    
    return dist_matrix, model_names

def pltDendrogram(dist_matrix, model_names, dataset, method='average', correlation='pearson', extension='png', dpi=100, show=True):
    save_folder = f'dataStorage/processedResults/dendrograms/{method}'
    
    condensed_dist = squareform(dist_matrix)
    
    Z = linkage(condensed_dist, method=method)
    
    coph = cophenet(Z)
    
    plt.figure(figsize=(8, 10))
    dendrogram(
        Z,
        orientation='right',
        labels=model_names,
        leaf_rotation=0,
        leaf_font_size=14,
        color_threshold=None # Can be set to a float to color clusters at a specific depth
    )
    
    #plt.title(dataset, fontsize=18)
    label_font_size = 18
    plt.ylabel("Models", fontsize=label_font_size)
    plt.xlabel(f"Distance = (1 - Pearson {correlation})/2", fontsize=label_font_size)
    plt.xlim(0.0, 0.5)
    plt.xticks(np.arange(0.0, 0.5, 0.05))
    plt.tick_params('x', labelsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/right{dataset}.{extension}", format=extension, dpi=dpi)
    if show: plt.show()
    
    
    pearson, r = pearsonr(coph, condensed_dist)
    spearman, r = spearmanr(coph, condensed_dist)
    
    print(type(coph))
    
    json_data = {f'{dataset}': {'cophnet': coph.tolist(), 'pearson': pearson, 'spearman': spearman}}
    
    appendCoephData(f"{save_folder}/data.json", json_data)
        

if __name__ == "__main__":
    

    for (dataset, subset) in datasets:

        dt = dataset.replace('/', '-')

        csv_path = getJsonInfo(json_path=jsonInfoPath(), fields=["rsaData"])[0]
        csv_path = csv_path+f"/{dt}Data.csv"
        
        correlation = 'pearson'
        
        data = findInCsv(csv_path, ["dataset"], [f"{dt}({subset})"])

        df = dataFrameFromData(data, correlation)
        
        print(df)

        dist_matrix, model_names = distMatirxModelNames(df)
        pltDendrogram(dist_matrix, model_names, f"{dt}", correlation=correlation, method='average', extension='eps', show=False)

