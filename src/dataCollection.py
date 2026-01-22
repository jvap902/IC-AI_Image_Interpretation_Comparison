import torch
from .plot import *
from .extraction.featureExtraction import getFeatureTensors
from tqdm.auto import tqdm

modelOutputCsv_path = './dataStorage/model_output/modelOutput.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"

def getSavePath(modelc, dt_info, embedding : bool):
    if embedding:
        return f'./dataStorage/model_output/embedding/{modelc.name}_{modelc.weights}_{modelc.source}_{dt_info.name_w_subset}.pt'
    else:
        return f'./dataStorage/model_output/std_output/{modelc.name}_{modelc.weights}_{modelc.source}_{dt_info.name_w_subset}.pt'

def stdOutput(modelc, inputs):
    modelc.model(inputs)
    
    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)
    
    return data.float()

def getStdOutputTensors(modelc, inputs):
    data_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(modelc.val_loader, desc="Extracting Data"):
            inputs = inputs.to(device)
            if modelc.model:
                modelc.model.eval()
                data = stdOutput(inputs) # Extract features using the backbone
            else:
                data = inputs.cpu() # Extract raw inputs
                
            data_list.append(data)
            labels_list.append(labels.cpu())
            
    return torch.cat(data_list), torch.cat(labels_list)

def getModelStdOutput(modelc, dt_info):
    stdOutput = getStdOutputTensors(modelc, modelc.val_loader)
    
    torch.save(stdOutput, getSavePath(modelc, dt_info, False))    
    
def getModelEmbeddings(modelc, dt_info):
    data = getFeatureTensors(modelc.val_loader, modelc)
    
    torch.save(data, getSavePath(modelc, dt_info, True))
    
def writeModelOutputLine(modelc, dt_info):
    std_output_path = getSavePath(modelc, dt_info, False)
    embedding_path = getSavePath(modelc, dt_info, True)
    
    data = [modelc.name, modelc.weights, modelc.source, dt_info.name_w_subset, std_output_path, embedding_path]
    
    writeCsvLine(modelOutputCsv_path, data)    
    
def gatherAdditionalData(modelc, dt_info):
    getModelStdOutput(modelc, dt_info)
    
    getModelEmbeddings(modelc, dt_info)
    
    writeModelOutputLine(modelc, dt_info)
