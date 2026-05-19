import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from src import config
from src.codifications import *
from src.fileManagement.csvUtils import *
from src.fileManagement.jsonUtils import getJsonInfo

json_info_path = config.json_info_path
datasets = config.datasets
metrics = ['pearson', 'spearman']
output_folder = f"{getJsonInfo(json_info_path)["processedResults"]}/datasetCorrelations"

instances = config.instances

def MtoMDatasetCorrelation(metric, df_dict):
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
    

def MRSS(datasets_dfs: dict, metric='pearson'):
    
    mrss_df = pd.DataFrame(columns=["Model", "MRSS"])
    
    if metric == 'pearson':
        correlation = pearsonr
    elif metric == 'spearmanr':
        correlation = spearmanr
    
    for model in instances:
    
        model_str = modelCod(model[0], model[1], model[2])
    
        new_data = {"Model": model_str}
        
        model_corr = []
        model_arrays = np.zeros((len(config.datasets), len(config.instances)-1))
        
        for idx, (key, df) in enumerate(datasets_dfs.items()):
            dt_df = df.drop(model_str, axis=1)
            model_arrays[idx] = dt_df.loc[model_str]
            
        for i in range(len(config.datasets)):
            for j in range(i+1, len(config.datasets)):
                model_corr.append(correlation(model_arrays[i], model_arrays[j]))
        
        model_corr = np.array(model_corr)
        
        new_data["MRSS"] = model_corr.mean()
        
        mrss_df.loc[len(mrss_df)] = new_data
    
    return mrss_df

def DRC(df_dict, metric):
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

    names = dtPaperName(datasets)

    df.columns = names
    df.index = names
    
    return df