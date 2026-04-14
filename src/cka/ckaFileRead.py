import json
import numpy as np
import pandas as pd
from ..codifications import *

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
        
        val = value["CKA"][0][0]
        
        data_matrix[i][j] = val
        data_matrix[j][i] = val
        
        
    return pd.DataFrame(data_matrix, columns=getCods(), index=getCods())
    

if __name__ == "__main__":
    pass