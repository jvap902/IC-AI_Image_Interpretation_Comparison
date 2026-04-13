import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import seaborn
import pandas as pd
import ast
from typing import Tuple
from src.codifications import getModelTrainStr

def writeCsvLine(file_path, data):
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(data)

def findInCsv(file_path, params, values):
    if(len(params) != len(values)):
        raise ValueError("The number of parameters should be the as the number of values serched")
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = list(csv.DictReader(file))
        
        ans = []
        
        for row in reader:
            all_equal = True
            
            for idx, param in enumerate(params):
                if (str(values[idx]) != row[param]):
                    all_equal = False
                    break
            
            if all_equal:
                ans.append(row)
                
    return ans

def getStringIntArray(string):
    
    string = string[1:-1]
    ans = [int(num) for num in string.split(', ')]
    
    return ans

def getStringStrArray(string):    
    
    return ast.literal_eval(string)

def paramDataFrameFromCsv(csv_path, param, diagonal=1.0):
    
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = list(csv.DictReader(file))
        
        return dataFrameFromData(reader, param, diagonal=diagonal)
    
    return dataFrame

def dataFrameFromData(data, param='pearson', diagonal=1.0, codification=False):
    if not data:
        return np.array([])

    # 1. Identificar todos os modelos únicos para saber o tamanho da matriz
    raw_items = []
    for row in data:
        raw_items.append((row['fst_model_source'], row['first_model'], row['fst_weights']))
        raw_items.append((row['snd_model_source'], row['second_model'], row['snd_weights']))
    
    # dict.fromkeys para remover duplicatas preservando a ordem
    models = list(dict.fromkeys(raw_items))
    
    n = len(models)
    
    # Criar matriz preenchida com zeros (ou 1.0 na diagonal)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, diagonal)

    # Criar um mapeamento de nome do modelo para índice (0, 1, 2...)
    model_to_idx = {model: i for i, model in enumerate(models)}

    # 2. Preencher a matriz com os valores do CSV
    for row in data:
        m1 = (row['fst_model_source'], row['first_model'], row['fst_weights'])
        m2 = (row['snd_model_source'], row['second_model'], row['snd_weights'])
        val = np.float32(row[param])
        
        i, j = model_to_idx[m1], model_to_idx[m2]
        matrix[i][j] = val
        matrix[j][i] = val # Garante a simetria se o CSV tiver apenas um lado

    if codification:
        for idx, (s, m, w) in enumerate(models):
            models[idx] = getModelTrainStr(s, m, w)

    dataFrame = pd.DataFrame(matrix, columns=models, index=models)

    return dataFrame

if __name__ == '__main__':
    pass