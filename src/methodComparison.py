from .cka import *
from .rsa import *
from .fileManagement.csvUtils import *
from .fileManagement.jsonUtils import *
from .fileManagement.defaultPaths import *
import numpy as np
from scipy.stats import pearsonr

datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)] #apenas datasets utilizados no artigo    
info_path = jsonInfoPath()

def rsaCka(rsa_dir, cka_dir):
    for (name, sub) in datasets[0:1]:
        
        name_with_subset = f"{name.replace('/', '-')}({sub})"
        
        df_rsa = findInCsv(f"{rsa_dir}/{name.replace('/', '-')}Data.csv", ['dataset'], [name_with_subset])
        df_rsa = dataFrameFromData(df_rsa, param='pearson', codification=True)

        rsa_array = df_rsa.to_numpy()[np.triu_indices(27, 1)]
        
        df_cka = ckaFileRead.dfFromCkaJson(f"{cka_dir}/{name_with_subset}/results.json")

        cka_array = df_cka.to_numpy()[np.triu_indices(27, 1)]
    
        p, _ = pearsonr(rsa_array, cka_array)

        print(df_rsa.corrwith(df_cka)) # model-wise
        print("Correlação: ", p)

if __name__ == "__main__":
    dir_paths = getJsonInfo(json_path=info_path, fields=["rsaData", "ckaData"])
    rsa_dir = dir_paths[0]
    cka_dir = dir_paths[1]
    
    rsaCka(rsa_dir, cka_dir)
