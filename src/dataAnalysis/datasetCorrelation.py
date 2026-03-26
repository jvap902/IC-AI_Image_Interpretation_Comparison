from ..plot import *
from typing import List,Tuple
from .codifications import *
from seaborn import heatmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ..fileSystem.fileSystem import updateJson
import json

results_folder = "./dataStorage/results"
datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)]
metric = 'spearman'
output_folder = "ztempData/datasetCorrelations"

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


def valueToColor(value):
    if value > 0.8:
        return "white"
    if value > 0.6:
        return "light_orange"
    if value > 0.4:
        return "orange"
    if value > 0.2:
        return "red"
    if value > 0.0:
        return "pink"
    if value > -0.2:
        return "purple"
    
    return "black"

def dtNameToAcr(n):
    match n:
        case 'imagenet-sketch':
            return 'sk'
        case 'cifar10':
            return 'cf'
        case 'fgvc-aircraft':
            return 'air'
        case 'ILSVRC/imagenet-1k':
            return 'ik'
        case _:
            raise


def savePkl(df, pkl_path, field_name, metric=metric, json_path=f"{output_folder}/pklPaths.json"):
    
    df.to_pickle(pkl_path)    
    
    with open(json_path, "r+") as f:
        json_data = json.load(f)
        
    json_data[metric][field_name] = pkl_path
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
        

def loadPkls(field_name, metric=metric, json_path=f"{output_folder}/pklPaths.json"):
    with open(json_path, 'r+') as f:
        json_data = json.load(f)
    
    pkl_path = json_data[metric][field_name]
    
    df = pd.read_pickle(pkl_path)
    
    return df    


def dtCorrelationHeatmaps(dic):
    for e in instances:
        df = pd.DataFrame(dic[e])

        names = dtNameSubset(datasets)

        df.columns = names
        df.index = names

        #print(df)
        plt.figure(figsize=(10, 8))
            
        heatmap(df, vmin=-0.5, vmax=1.0)
        
        model_str = getModelTrainStr(e[0], e[1], e[2])
        
        pkl_path = f"{output_folder}/pkls/{model_str}.pkl"

        savePkl(df, pkl_path, model_str)
        
        plt.title(str(e))
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{metric}/{model_str}.png")
        #plt.show()


def dtCorrelationGrouping(dic):
    
    #sk = sketch, cf = cifar, ik = imagenet-1k, air = aircraft
    
    similar_bhvs = dict() #dicionario de dicionários de listas de modelos com comportamentos semelhantes
    similar_bhvs["abstract"]=[]
    similar_bhvs["sk_cf"]=dict()
    similar_bhvs["sk_air"]=dict()
    similar_bhvs["sk_ik"]=dict()
    similar_bhvs["cf_air"]=dict()
    similar_bhvs["cf_ik"]=dict()
    similar_bhvs["air_ik"]=dict()
    
    for key, value in similar_bhvs.items():
        if key=="abstract":
            continue
        value["black"] = []
        value["purple"] = []
        value["pink"] = []
        value["red"] = []
        value["orange"] = []
        value["light_orange"] = []
        value["white"] = []
    
    acronyms = [dtNameToAcr(dt[0]) for dt in datasets]
    
    for e in instances:
        for i, d1 in enumerate(acronyms):
            for j, d2 in enumerate(acronyms[i+1:]):
                val = dic[e][i][j+i+1]
                color = valueToColor(val)
                similar_bhvs[f"{d1}_{d2}"][color].append((getModelTrainStr(e[0], e[1], e[2]), val))
                
    for key, value in similar_bhvs.items():
        if key=="abstract":
            continue
        
        abst = ', '.join([f"{color}: {len(v)}" for color, v in value.items()])
        abst = f"{key} -> " + abst
        
        similar_bhvs['abstract'].append(abst)
    
    with open(f'{output_folder}/{metric}Data.json', 'w') as f:
        json.dump(similar_bhvs, f, indent=4)
        
    return similar_bhvs


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
        
    dtCorrelationHeatmaps(dic)
    
    #similar_bhv = dtCorrelationGrouping(dic)
        
        
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

            if metric == 'pearson':
                val, _ = pearsonr(dt1, dt2)
            elif metric == 'spearman':
                val, _ = spearmanr(dt1, dt2)
            else:
                raise
            
            data[i][j] = val
            data[j][i] = val

    _ = np.fill_diagonal(data, 1.0)


    df = pd.DataFrame(data)

    names = dtNameSubset(datasets)

    df.columns = names
    df.index = names

    #print(df)
    
    pkl_path = f"{output_folder}/pkls/{metric}.pkl"

    savePkl(df, pkl_path, "main")
    
    plt.figure(figsize=(10, 8))
    
    heatmap(df, vmin=-0.5, vmax=1.0)
    
    plt.title(f"Correlação de {metric} entre datasets")
    plt.tight_layout()
    plt.savefig(f"ztempData/datasetCorrelations/{metric}.png")
    #plt.show()
    
    return df


def MRSS(model, json_path=f"{output_folder}/mrss.json"):
    
    model_str = getModelTrainStr(model[0], model[1], model[2])
    
    
    # Fazer efetivamente os cálculos
    
    new_data = dict()
    new_data['pearson'] = {'avg': p_avg, 'med': p_med}
    new_data['spearman'] = {'avg': s_avg, 'med': s_med}
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    json_data[model_str] = new_data
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    


if __name__ == "__main__":
    MtoMDatasetCorrelation()
    generalDatasetCorrelation()