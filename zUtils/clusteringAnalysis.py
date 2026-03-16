from ..src.plot import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)] #apenas datasets utilizados no artigo

def distMatrix(df):
    dist_matrix = np.empty((27, 27))

    for idx, row in df.iterrows():
        return

    condensed_dist = squareform(dist_matrix)
    
    # 3. Perform Hierarchical Clustering
    # 'ward' minimizes variance, 'complete' finds the maximum distance between clusters
    Z = linkage(condensed_dist, method='complete')
    
    # 4. Plotting
    plt.figure(figsize=(12, 7))
    dendrogram(
        Z,
        labels=model_names,
        leaf_rotation=45,
        leaf_font_size=12,
        color_threshold=None # Can be set to a float to color clusters at a specific depth
    )
    
    plt.title("Dendograma", fontsize=16)
    plt.ylabel("Distance (1 - Spearman Correlation)")
    plt.xlabel("Models")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#dist = d(i,j) = (1-Corr(i,j))/2



        

if __name__ == "__main__":
    

    for (dataset, subset) in datasets:

        dt = dataset.replace('/', '-')

        csv_path = f'./ztempData/{dt}Data.csv'

        df = paramDataFrameFromCsv(csv_path, 'pearson')

        dist_matrix = distMatrix(df)

