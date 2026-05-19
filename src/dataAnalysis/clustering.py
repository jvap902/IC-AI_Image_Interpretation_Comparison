import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cophenet
from src import config

cods = config.cods

def distMatrix(df):
    #dist = d(i,j) = (1-Corr(i,j))/2
    dist_matrix = np.empty([len(cods), len(cods)])

    for idx, (key, row) in enumerate(df.iterrows()):
        for i, cod in enumerate(cods):
            dist_matrix[idx][i] = (1 - row[cod])/2
            dist_matrix[i][idx] = dist_matrix[idx][i]
        
    return dist_matrix
        

def linkData(dist_matrix, method='average'):
    condensed_dist = squareform(dist_matrix)
    
    Z = linkage(condensed_dist, method=method)
    
    coph = cophenet(Z)
    
    pearson, r = pearsonr(coph, condensed_dist)
    spearman, r = spearmanr(coph, condensed_dist)
        
    dic = {'cophnet': coph.tolist(), 'pearson': pearson, 'spearman': spearman, 'linkage': Z}
    
    return dic