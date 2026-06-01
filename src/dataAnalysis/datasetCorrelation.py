import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from src import config
from src.codifications import *

instances = config.instances
cods = config.cods

def modelWiseDRC(df_dict, metric, datasets):
    dic = {}
    for e in cods:
        dic[e] = np.zeros((len(datasets), len(datasets)))
    #data = np.zeros((len(datasets), len(datasets)))

    for i in range(len(datasets)):
        for j in range(i):
            dt1_name = dtNameSubset(datasets[i])
            dt2_name = dtNameSubset(datasets[j])

            df1 = df_dict[dt1_name]
            df2 = df_dict[dt2_name]

            df_corr = df1.corrwith(df2, method=metric)

            for idx, e in enumerate(cods):
                val = df_corr[cods[idx]]

                dic[e][i][j] = val
                dic[e][j][i] = val

    for e in instances:
        _ = np.fill_diagonal(dic[e], 1.0)
        
    return dic
    

def MRSS(datasets_dfs: dict, datasets=config.datasets, metric='pearson'):
    
    mrss_df = pd.DataFrame(columns=["model", "mrss"])
    
    if metric == 'pearson':
        correlation = pearsonr
    elif metric == 'spearmanr':
        correlation = spearmanr
    
    for model in instances:
    
        model_str = modelCod(model[0], model[1], model[2])
    
        new_data = {"model": model_str}
        
        model_corr = []
        model_arrays = np.zeros((len(datasets), len(config.instances)-1))
        
        for idx, (key, df) in enumerate(datasets_dfs.items()):
            dt_df = df.drop(model_str, axis=1)
            model_arrays[idx] = dt_df.loc[model_str]
            
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                model_corr.append(correlation(model_arrays[i], model_arrays[j]))
        
        model_corr = np.array(model_corr)
        
        new_data["mrss"] = model_corr.mean()
        
        mrss_df.loc[len(mrss_df)] = new_data
    
    return mrss_df

def DRC(df_dict, datasets, metric='pearson'):
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