from .cka import *
from .rsa import *
from .fileManagement.csvUtils import *
from .fileManagement.jsonUtils import *
from .fileManagement.defaultPaths import *

info_path = jsonInfoPath()
dir_paths = getJsonInfo(["rsaData", "ckaData"], json_path=info_path)
rsa_dir = dir_paths[0]
cka_dir = dir_paths[1]

if __name__ == "__main__":
    
    datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)] #apenas datasets utilizados no artigo    
    
    for (name, sub) in datasets[1:2]:
        
        name_with_subset = f"{name.replace('/', '-')}({sub})"
        
        df_rsa = findInCsv(f"{rsa_dir}/{name.replace('/', '-')}Data.csv", ['dataset'], [name_with_subset])
        df_rsa = dataFrameFromData(df_rsa, param='pearson', codification=True)
        
        df_cka = ckaFileRead.dfFromCkaJson(f"{cka_dir}/{name_with_subset}/results.json")
    
        print(df_rsa.corrwith(df_cka))