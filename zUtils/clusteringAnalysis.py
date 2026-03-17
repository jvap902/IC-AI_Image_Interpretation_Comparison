from src.plot import *
from .codifications import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)] #apenas datasets utilizados no artigo

instances = [
        ('huggingface', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'DEFAULT'),
        ('huggingface', 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'DEFAULT'),
        ('clip', 'ViT-B/32', 'DEFAULT'),
        ('clip', 'ViT-B/16', 'DEFAULT'),
        ('clip', 'ViT-L/14', 'DEFAULT'),
        ('open_clip', 'ViT-B-32-256', 'DEFAULT'),
        ('open_clip', 'ViT-B-16', 'DEFAULT'),
        ('open_clip', 'ViT-L-14', 'DEFAULT'),
        ('torchvision', 'resnet18', 'IMAGENET1K_V1'),
        ('torchvision', 'resnet50', 'IMAGENET1K_V1'),
        ('torchvision', 'resnet152', 'IMAGENET1K_V1'),
        ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V1'),
        ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V2'),
        ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_SWAG_E2E_V1'),
        ('torchvision', 'regnet_y_32gf', 'IMAGENET1K_V2'),
        ('torchvision', 'vit_b_16', 'IMAGENET1K_V1'),
        ('torchvision', 'vit_b_16', 'IMAGENET1K_SWAG_E2E_V1'),
        ('torchvision', 'vit_l_16', 'IMAGENET1K_V1'),
        ('torchvision', 'vit_h_14', 'IMAGENET1K_SWAG_E2E_V1'),
        ('torchvision', 'maxvit_t', 'IMAGENET1K_V1'),
        ('torchvision', 'convnext_tiny', 'IMAGENET1K_V1'),
        ('torchvision', 'convnext_base', 'IMAGENET1K_V1'),
        ('torchvision', 'swin_t', 'IMAGENET1K_V1'),
        ('torchvision', 'swin_v2_t', 'IMAGENET1K_V1'),
        ('torchvision', 'efficientnet_b0', 'IMAGENET1K_V1'),
        ('torchvision', 'efficientnet_b4', 'IMAGENET1K_V1'),
        ('torchvision', 'efficientnet_b7', 'IMAGENET1K_V1')   
    ]

def distMatirxModelNames(df):
    #dist = d(i,j) = (1-Corr(i,j))/2
    dist_matrix = np.empty([27, 27])
    
    model_names = set()

    for idx, (key, row) in enumerate(df.iterrows()):
        model_names.add(key)
        
        for i, instance in enumerate(instances):
            dist_matrix[idx][i] = (1 - row[instance])/2
    
    model_names = list(model_names)
    
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

