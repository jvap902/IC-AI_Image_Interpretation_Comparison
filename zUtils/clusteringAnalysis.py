from src.plot import *
from .codifications import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)] #apenas datasets utilizados no artigo

instances = getInstances()

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

def pltDendogram(dist_matrix, model_names, dataset):
    condensed_dist = squareform(dist_matrix)
    
    Z = linkage(condensed_dist, method='complete')
    
    plt.figure(figsize=(15, 9))
    dendrogram(
        Z,
        labels=model_names,
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=None # Can be set to a float to color clusters at a specific depth
    )
    
    plt.title(dataset, fontsize=16)
    plt.ylabel("Distance = (1 - Pearson Correlation)/2")
    plt.xlabel("Models")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
        

if __name__ == "__main__":
    

    for (dataset, subset) in datasets:

        dt = dataset.replace('/', '-')

        csv_path = f'./dataStorage/results/{dt}Data.csv'
        
        data = findInCsv(csv_path, ["dataset"], [f"{dt}({subset})"])

        df = dataFrameFromData(data, 'pearson')
        
        print(df)

        dist_matrix, model_names = distMatirxModelNames(df)
        pltDendogram(dist_matrix, model_names, f"{dt}({subset})")

