from ..plot import *
from typing import List,Tuple
from .codifications import *
from seaborn import heatmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

results_folder = "./dataStorage/results"
datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)]
metric = 'spearman'

instances = getInstances()

def dtNameSubset(dt: Tuple[str, int] | List[Tuple[str, int]]):
    if isinstance(dt, tuple):
        name, subset = dt
        return f"{name.replace('/', '-')}({subset})"
    
    # Otherwise, treat it as a list of tuples
    return [f"{d[0].replace('/', '-')}({d[1]})" for d in dt]

def getDataFrames(results_folder=results_folder, datasets: List[Tuple[str, int]] = datasets):

    df_dict = {}

    for dt in datasets:
        dt_name_w_subset = dtNameSubset(dt)

        csv_path = f"{results_folder}/{dt[0].replace('/', '-')}Data.csv"

        data = findInCsv(csv_path, ['dataset'], [dt_name_w_subset])

        df_dict[dt_name_w_subset] = dataFrameFromData(data, metric)

    return df_dict


def MtoMDatasetCorrelation():
    df_dict = getDataFrames()

    dic = {}
    for e in instances:
        dic[e] = np.zeros((len(datasets), len(datasets)))
    #data = np.zeros((len(datasets), len(datasets)))

    for i in range(len(datasets)):
        for j in range(i):
            dt1_name = dtNameSubset(datasets[i])
            dt2_name = dtNameSubset(datasets[j])

            df1 = df_dict[dt1_name]
            df2 = df_dict[dt2_name]

            correlations = df1.corrwith(df2, method=metric)

            for idx, e in enumerate(instances):
                val = correlations[instances[idx]]

                dic[e][i][j] = val
                dic[e][j][i] = val

    for e in instances:
        _ = np.fill_diagonal(dic[e], 1.0)

    dfs = {}
    for e in instances:
        df = pd.DataFrame(dic[e])

        names = dtNameSubset(datasets)

        df.columns = names
        df.index = names

        #print(df)

        plt.figure(figsize=(10, 8))
        
        heatmap(df, vmin=-0.5, vmax=1.0)
        
        plt.title(str(e))
        plt.tight_layout()
        plt.savefig(f"ztempData/datasetCorrelations/{metric}/{getModelTrainStr(e[0], e[1], e[2]).replace(', ', '-')}.png")
        #plt.show()
        
def generalDatasetCorrelation():
    df_dict = getDataFrames()

    data = np.zeros((len(datasets), len(datasets)))

    for i in range(len(datasets)):
        dt1_name = dtNameSubset(datasets[i])
        df1 = df_dict[dt1_name]
        
        for j in range(i):
            dt2_name = dtNameSubset(datasets[j])
            df2 = df_dict[dt2_name]
            
            dt1 = df1.to_numpy()[np.triu_indices(n=len(instances), k=1, m=len(instances))]
            dt2 = df2.to_numpy()[np.triu_indices(n=len(instances), k=1, m=len(instances))]

            p, _ = pearsonr(dt1, dt2)
            
            data[i][j] = p
            data[j][i] = p

    _ = np.fill_diagonal(data, 1.0)


    df = pd.DataFrame(data)

    names = dtNameSubset(datasets)

    df.columns = names
    df.index = names

    #print(df)

    plt.figure(figsize=(10, 8))
    
    heatmap(df, vmin=-0.5, vmax=1.0)
    
    plt.title(f"Correlação de {metric} entre datasets")
    plt.tight_layout()
    plt.savefig(f"ztempData/datasetCorrelations/{metric}.png")
    plt.show()

if __name__ == "__main__":
    generalDatasetCorrelation()