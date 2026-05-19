
import pandas as pd
import numpy as np
from fileManagement.jsonUtils import getJsonInfo
from codifications import codToInstance
import config

def verifyIntegrity(data):    
    keys = set(data.keys())
    
    cods = config.cods
    
    c = True
    
    for i in range(len(cods)):
        for j in range(i+1, len(cods)):
            if(f"{cods[i]} {cods[j]}" not in keys):
                c = False
                print(f"ERRO - {cods[i]} {cods[j]} não encontrado")
    
    return c

def resultsDataFrame(json_path, verify=True):
    json_data = getJsonInfo(json_path)
    
    if not verifyIntegrity(json_data) and verify:
        raise ValueError("Nem todos os dados CKA estão no json")
    
    data_matrix = np.zeros((len(config.instances), len(config.instances)))
    np.fill_diagonal(data_matrix, 1)
    
    for key, value in json_data.items():
        cod1, cod2 = key.split(' ')
        
        n1, l1 = int(cod1[:-1]), cod1[-1]
        n2, l2 = int(cod2[:-1]), cod2[-1]
        
        i, j = codToInstance(n1, l1)[0], codToInstance(n2, l2)[0]
        
        data_matrix[i][j] = value
        data_matrix[j][i] = value

    return pd.DataFrame(data_matrix, columns=config.cods, index=config.cods)


def getCkaData(dataset, subset):
    dataset = dataset.replace('/', '-')

    dir = getJsonInfo(config.json_info_path, ["ckaData"])[0]+f"{dataset}({subset})"

    json_path = dir+"results.json"

    return resultsDataFrame(json_path, verify=True)