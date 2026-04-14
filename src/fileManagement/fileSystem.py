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
    
    # --- Folders ---
    
    os.makedirs("data", exist_ok=True)
    
    for key, value in paths.items(): #garante que todas as pastas em paths existem
        os.makedirs(value, exist_ok=True)

    ds = paths["output_dir"]
    mo = paths["model_output"]
    
    # --- Files ---
    
    model_params = 'model,model_source,model_weights,dataset'
    
    files_paths = dict()
    
    files_paths = createFile(paths["dissimilarity_folder"]+'/cosineDissimilarity.csv', model_params+',path\n', files_paths)
    
    files_paths = createFile('classes.csv', 'dataset,subset,num_classes,num_images,train_classes,validation_classes\n', files_paths)
    
    files_paths = createFile(ds+'/datasetClasses.csv', 'dataset,subset,num_classes,num_images,train_classes,validation_classes\n', files_paths)
    
    files_paths = createFile(ds+'/selectedIndices.csv', 'file_name,train_indices,validation_indices\n', files_paths)
    
    files_paths = createFile(ds+'/validation_results.csv', model_params+'accuracy\n', files_paths)
    
    files_paths = createFile(mo+'/modelOutput.csv', model_params+'(subset),std_output_path,embedding_path\n', files_paths)
    
    files_paths = createFile(outputFile, 'total_images,num_classes,fst_model_source,first_model,fst_weights,snd_model_source,second_model,snd_weights,first_model_accuracy,second_model_accuracy,spearman,pearson,dataset\n', files_paths)
    
    paths.update(files_paths)
    
    writeJson(json_path=jsonInfoPath(), dic=paths)

    
def createFile(file_path, content, files_paths: Dict = None):
    
    if files_paths != None:
        key = file_path.split('/')[-1]
        key = key.split('.')[-2]
        
        files_paths[key] = file_path
    
    if os.path.isfile(file_path):
        print("Arquivo já existente")
    else:
        print("Arquivo não existente, criando novo")
        with open(file_path, mode="a", newline='', encoding='utf-8') as f:
            f.write(content)

    return files_paths

if __name__ == '__main__':
    pass