import numpy as np
import pandas as pd
from src.fileManagement.jsonUtils import getJsonInfo
from src.fileManagement.csvUtils import findInCsv
from src.codifications import modelCod
from src import config

def dictDataFrame(data, param='pearson', diagonal=1.0, codify=False):
    if not data:
        return np.array([])

    models = config.instances
    
    n = len(models)
    
    # Criar matriz preenchida com zeros (ou 1.0 na diagonal)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, diagonal)

    # 2. Preencher a matriz com os valores do CSV
    for row in data:
        m1 = (row['fst_model_source'], row['first_model'], row['fst_weights'])
        m2 = (row['snd_model_source'], row['second_model'], row['snd_weights'])
        val = np.float32(row[param])
        
        i, j = models.index(m1), models.index(m2)
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