import numpy as np
import pandas as pd
from src.fileManagement.jsonUtils import getJsonInfo
from src.fileManagement.csvUtils import findInCsv
from src.codifications import modelCod
from src import config

def dictDataFrame(data, param='pearson', diagonal=1.0, codify=False):
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

    if codify:
        for idx, (s, m, w) in enumerate(models):
            models[idx] = modelCod(s, m, w)

    dataFrame = pd.DataFrame(matrix, columns=models, index=models)

    return dataFrame

def getRsaData(dataset, subset, param='pearson', codify=True):
    dataset = dataset.replace('/', '-')

    dir = getJsonInfo(config.json_info_path, ["rsaData"])[0]
    csv_path = dir+f"/{dataset}Data.csv"

    data = findInCsv(csv_path, ["dataset"], [f"{dataset}({subset})"])
    df = dictDataFrame(data, param=param, codify=codify)

    return df