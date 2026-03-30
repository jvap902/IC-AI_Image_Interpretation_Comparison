import csv
import numpy as np
from src.plot import *
from pandas import DataFrame
from src.codifications import getModelTrainStr

datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)] #apenas datasets utilizados no artigo

def getAverages(data):
    
    p_array = []
    s_array = []
    
    for row in data:
        p_array.append(np.float32(row['pearson']))
        s_array.append(np.float32(row['spearman']))
        
    pearson = np.array(p_array)
    spearman = np.array(s_array)
    
    return pearson.mean(), spearman.mean()

def getMinMaxStr(data) -> str:
    p_min = 10000
    s_min = 10000
    p_max = -10000
    s_max = -10000
    
    for row in data:
        
        p_value = np.float32(row['pearson'])
        s_value = np.float32(row['spearman'])
        
        if p_value < p_min:
            pearson_min, p_min = f"{row['first_model']}, {row['second_model']} -> {p_value}", p_value
        
        if s_value < s_min:
            spearman_min, s_min = f"{row['first_model']}, {row['second_model']} -> {s_value}", s_value
        
        if p_value > p_max:
            pearson_max, p_max = f"{row['first_model']}, {row['second_model']} -> {p_value}", p_value
            
        if s_value > s_max:
            spearman_max, s_max = f"{row['first_model']}, {row['second_model']} -> {s_value}", s_value
            
    return pearson_min, spearman_min, pearson_max, spearman_max

def getMinMaxModelAvg(data):
    model_pearson = dict()
    model_spearman = dict()
    
    for row in data:
        current = [row['first_model'], row['second_model']]
        
        for m in current:
            if not (m in model_pearson):
                model_pearson[m] = []
                model_spearman[m] = []
            
            model_pearson[m].append(np.float32(row['pearson']))
            model_spearman[m].append(np.float32(row['spearman']))
            
    for key, value in model_pearson.items():
        model_pearson[key] = np.array(value)
    
    for key, value in model_spearman.items():
        model_spearman[key] = np.array(value)
        
    min_p, min_s = (10000, "a"), (10000, "a")
    max_p, max_s = (-10000, "a"), (-10000, "a")
    
    for key, value in model_pearson.items():
        
        if value.mean() < min_p[0]:
            min_p = (value.mean(), key)
        
        if value.mean() > max_p[0]:
            max_p = (value.mean(), key)
    
    for key, value in model_spearman.items():
        
        if value.mean() < min_s[0]:
            min_s = (value.mean(), key)
        
        if value.mean() > max_s[0]:
            max_s = (value.mean(), key)
    
    return (min_p, min_s), (max_p, max_s)

def buildFromBeginning(txt):
    for (dataset, subset) in datasets:
        dt_name = dataset.replace('/','-')
        csv_name = (f"./dataStorage/results/{dt_name}Data.csv")
        
        data = findInCsv(csv_name, ['dataset'], [f'{dt_name}({subset})'])
        
        pearson_average, spearman_average = getAverages(data)
        
        p_min, s_min, p_max, s_max = getMinMaxStr(data)
        
        min_avg, max_avg = getMinMaxModelAvg(data)
                    
        txt.write(f"--- Results for {dataset}({subset}) ---\n")
        txt.write(f"Pearson average: {pearson_average}\n")
        txt.write(f"Spearman average: {spearman_average}\n")
        txt.write(f"Pearson min value: {p_min} | Spearman min value: {s_min}\n")
        txt.write(f"Pearson max value: {p_max} | Spearman max value: {s_max}\n")
        txt.write(f"Pearson min model avg: {min_avg[0][1]} -> {min_avg[0][0]} | Spearman min model avg: {min_avg[1][1]} -> {min_avg[1][0]}\n")
        txt.write(f"Pearson max model avg: {max_avg[0][1]} -> {max_avg[0][0]} | Spearman max model avg: {max_avg[1][1]} -> {max_avg[1][0]}\n")
        
        txt.write("\n")

        ModelModelAvgStd(txt)

def dataAsDict():
    info = dict()

    for (dataset, subset) in datasets:
        dt_name = dataset.replace('/','-')
        csv_name = (f"./dataStorage/results/{dt_name}Data.csv")
            
        data = findInCsv(csv_name, ['dataset'], [f'{dt_name}({subset})'])

        info[f'{dt_name}({subset})'] = dict()

        for row in data:
            first_model = row['first_model']
            second_model = row['second_model']
            pearson = np.float32(row['pearson'])
            spearman = np.float32(row['spearman'])

            info[f'{dt_name}({subset})'][frozenset({first_model, second_model})] = (pearson, spearman)

    return info

def ModelModelAvgStd(txt):

    all_dfs = []

    for (dataset, subset) in datasets:
        dt_name = dataset.replace('/','-')

        data = findInCsv(f"./dataStorage/results/{dt_name}Data.csv", ['dataset'], [f'{dt_name}({subset})'])

        df = dataFrameFromData(data, 'pearson')

        all_dfs.append(df.to_numpy())
            
    stacked_data = np.stack(all_dfs)
    
    grouped_data = [
        [stacked_data[:, r, c] for c in range(stacked_data.shape[2])]
        for r in range(stacked_data.shape[1])
    ]
    
    processed_data = [[""] * 27 for _ in range(27)] 
    
    for i, row in enumerate(grouped_data):
        for j, col in enumerate(row):
            processed_data[i][j] = f"{np.round(np.array(col).mean(), 5)} +- {np.round(np.std(col), 5)}"
        
    labels = [getModelTrainStr(src, m, t) for src, m, t in df.columns]
    
    master_df = pd.DataFrame(processed_data, index=labels, columns=labels)
        
    print(master_df)

    txt.write("\n\n")
    txt.write(master_df.to_string())

    


if __name__ == "__main__":
    
    with open("dataAnalysis.txt", "a", encoding="utf-8") as txt:
        
        ModelModelAvgStd(txt)