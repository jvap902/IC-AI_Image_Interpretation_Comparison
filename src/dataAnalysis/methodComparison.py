import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from src import config
from src.fileManagement import csvUtils, ckaFileHandler, jsonUtils

info_path = config.json_info_path

def rsaCka(rsa_dir, cka_dir, datasets):
    for (name, sub) in datasets:
        
        name_with_subset = f"{name.replace('/', '-')}({sub})"
        
        df_rsa = pd.dataFrame(csvUtils.findInCsv(f"{rsa_dir}/{name.replace('/', '-')}Data.csv", ['dataset'], [name_with_subset]))
        df_rsa = df_rsa[["pearson"]]

        rsa_array = df_rsa.to_numpy()[np.triu_indices(27, 1)]
        
        df_cka = ckaFileHandler.dfFromCkaJson(f"{cka_dir}/{name_with_subset}/results.json")

        cka_array = df_cka.to_numpy()[np.triu_indices(27, 1)]
    
        p, _ = pearsonr(rsa_array, cka_array)

        print(df_rsa.corrwith(df_cka)) # model-wise
        print("Correlação: ", p)

if __name__ == "__main__":
    dir_paths = jsonUtils.getJsonInfo(json_path=info_path, fields=["rsaData", "ckaData"])
    rsa_dir = dir_paths[0]
    cka_dir = dir_paths[1]

    dt = config.datasets

    dt = [dt[2]]
    
    rsaCka(rsa_dir, cka_dir, dt)
