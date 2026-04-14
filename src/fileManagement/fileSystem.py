import os
import json
from typing import Dict
from .defaultPaths import *
from .jsonUtils import *

def makeFileSystem(outputFile):
    
    paths = {
        "output_dir": "dataStorage",
        "cache": "cache"
    }
    paths["model_output"] = f"{paths["output_dir"]}/model_output"
    paths.update(
        {
            "dissimilarity_folder": f"{paths["cache"]}/dissimilarity_arrays",
            "cka_matrices_folder": f"{paths["cache"]}/cka_matrices",
            "embedding_folder": f"{paths["model_output"]}/embedding",
            "std_output_folder": f"{paths["model_output"]}/std_output",
            "rsaData": f"{paths["output_dir"]}/rsaData",
            "ckaData": f"{paths["output_dir"]}/ckaData",
            "processedResults": f"{paths["output_dir"]}/processedResults"
        }
    )
    
    writeJson(json_path=jsonInfoPath(), dic=paths)
    
    # --- Folders ---
    
    os.makedirs("data", exist_ok=True)
    
    for key, value in paths.items(): #garante que todas as pastas em paths existem
        os.makedirs(value, exist_ok=True)

    ds = paths["output_dir"]
    mo = paths["model_output"]
    
    # --- Files ---
    
    model_params = 'model,model_source,model_weights,dataset'
    
    createFile(paths["dissimilarity_folder"]+'/cosineDissimilarity.csv', model_params+',path\n')
    
    createFile(ds+'/datasetClasses.csv', 'dataset,subset,num_classes,num_images,train_classes,validation_classes\n')
    
    createFile(ds+'/selectedIndices.csv', 'file_name,train_indices,validation_indices\n')
    
    createFile(ds+'/validation_results.csv', model_params+'accuracy\n')
    
    createFile(mo+'/modelOutput.csv', model_params+'(subset),std_output_path,embedding_path\n')
    
    createFile(outputFile, 'total_images,num_classes,fst_model_source,first_model,fst_weights,snd_model_source,second_model,snd_weights,first_model_accuracy,second_model_accuracy,spearman,pearson,dataset\n')

    
def createFile(file_path, content):
    
    if os.path.isfile(file_path):
        print("Arquivo já existente")
    else:
        print("Arquivo não existente, criando novo")
        with open(file_path, mode="a", newline='', encoding='utf-8') as f:
            f.write(content)

if __name__ == '__main__':
    pass