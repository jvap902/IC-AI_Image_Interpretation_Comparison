import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from src import config

instances = config.instances

def distMatrix(df):
    #dist = d(i,j) = (1-Corr(i,j))/2
    dist_matrix = np.empty([len(instances), len(instances)])

    for idx, (key, row) in enumerate(df.iterrows()):
        for i, instance in enumerate(instances):
            dist_matrix[idx][i] = (1 - row[instance])/2
            dist_matrix[i][idx] = dist_matrix[idx][i]
        
    return dist_matrix

def pltDendrogram(Z, labels, title=None, ylabel=None, xlabel=None, dpi=100, show=True, save_path = None):
    
    plt.figure(figsize=(8, 10))
    plt.rcParams['lines.linewidth'] = 2.5
    dendrogram(
        Z,
        orientation='right',
        labels=labels,
        leaf_rotation=0,
        leaf_font_size=14,
        color_threshold=None # Can be set to a float to color clusters at a specific depth
    )
    
    label_font_size = 18
    plt.xlim(0.0, 0.5)
    plt.xticks(np.arange(0.0, 0.5, 0.05))
    plt.tick_params('x', labelsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.2)
    if title: plt.title(title, fontsize=label_font_size)
    if ylabel: plt.ylabel(ylabel, fontsize=label_font_size)
    if xlabel: plt.xlabel(xlabel, fontsize=label_font_size)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=dpi)
    if show: plt.show()
    
    return
        

def linkData(dist_matrix, method='average'):
    condensed_dist = squareform(dist_matrix)
    
    Z = linkage(condensed_dist, method=method)
    
    coph = cophenet(Z)
    
    pearson, r = pearsonr(coph, condensed_dist)
    spearman, r = spearmanr(coph, condensed_dist)
    
    print(type(coph))
    
    dic = {'cophnet': coph.tolist(), 'pearson': pearson, 'spearman': spearman, 'linkage': Z}
    
    return dic