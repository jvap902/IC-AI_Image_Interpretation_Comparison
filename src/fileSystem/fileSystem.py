import os
import json
from typing import Dict

default_json_path = 'src/fileSystem/info.json'

def makeFileSystem(outputFile):
    
    # --- Folders ---

    os.makedirs("data", exist_ok=True)

    os.makedirs("datasetCache", exist_ok=True)
    
    ds = "dataStorage"
    os.makedirs(ds, exist_ok=True) # Ensure dataStorage folder exists

    os.makedirs(ds+"ckaFeatures", exist_ok=True)

    res = ds+"/results"
    os.makedirs(res, exist_ok=True)
    
    os.makedirs(ds+"/dissimilarity_arrays", exist_ok=True)
    
    mo = ds+"/model_output"
    os.makedirs(mo, exist_ok=True)
    
    os.makedirs(mo+"/embedding", exist_ok=True)
    
    os.makedirs(mo+"/std_output", exist_ok=True)    
    
    
    # --- Files ---
    
    model_params = 'model,model_source,model_weights,dataset'
    
    createFile(ds+'/cosineDissimilarity.csv', model_params+',path\n')
    
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

def updateJson(fields, values, json_path=default_json_path):
    with open(json_path, "r+") as f:
        json_data = json.load(f)
        
        for idx, field in enumerate(fields):
            json_data[field] = values[idx]
        
        f.seek(0)
        json.dump(json_data, f, indent=4)
        f.truncate()

def getJsonInfo(fields=[], json_path=default_json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    
    if len(fields) == 0:
        return json_data
    
    data = []

    for field in fields:
        data.append(json_data[field])
        
    return data

if __name__ == '__main__':
    pass