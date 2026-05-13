import json
import numpy as np
import pandas as pd
from src import config
from src.codifications import codToInstance

def dfFromCkaJson(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    data_matrix = np.zeros((27, 27))
    np.fill_diagonal(data_matrix, 1)
    
    for key, value in json_data.items():
        cod1, cod2 = key.split(' ')
        
        n1, l1 = int(cod1[:-1]), cod1[-1]
        n2, l2 = int(cod2[:-1]), cod2[-1]
        
        i, j = codToInstance(n1, l1)[0], codToInstance(n2, l2)[0]
        
        data_matrix[i][j] = value
        data_matrix[j][i] = value
        
        
    return pd.DataFrame(data_matrix, columns=config.cods, index=config.cods)
    

if __name__ == "__main__":
    pass