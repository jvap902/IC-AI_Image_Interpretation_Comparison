import torch
from .plot import *
from .extraction.featureExtraction import getFeatureTensors
from tqdm.auto import tqdm
from .model.modelClass import Model
from .dataset.datasetClass import DtInfo
from .dataset.datasetUtils import getClasses
import clip

modelOutputCsv_path = './dataStorage/model_output/modelOutput.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"

def getSavePath(modelc, dt_info, embedding : bool):
    m_name = modelc.name.replace('/', '-')
    if embedding:
        return f'./dataStorage/model_output/embedding/{m_name}_{modelc.weights}_{modelc.source}_{dt_info.name_w_subset}.pt'
    else:
        return f'./dataStorage/model_output/std_output/{m_name}_{modelc.weights}_{modelc.source}_{dt_info.name_w_subset}.pt'

def stdOutputExtractor(modelc : Model, inputs):
    match modelc.source:
        case 'open_clip':
            return torch.tensor(modelc.model(inputs, text))
        
        case 'clip':
            text = clip.tokenize(getClasses(modelc.val_dataset)).to(device)
            logits_img, logits_txt = modelc.model(inputs, text)
            
            return torch.tensor(logits_img)
        
        case 'huggingface':
            if 'dinov3' in modelc.name:
                raise NotImplemented
            else:
                raise ValueError("Unsupported model")
            
        case _:
            return modelc.model(inputs)

def stdOutput(modelc : Model, inputs):
    
    data = stdOutputExtractor(modelc, inputs)
    
    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)
    
    return data.float()

def getStdOutputTensors(modelc : Model, loader):   
    data_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Extracting Data"):
            inputs = inputs.to(device)
            if modelc.model:
                modelc.model.eval()
                data = stdOutput(modelc, inputs) # Extract features using the backbone
            else:
                data = inputs.cpu() # Extract raw inputs
                
            data_list.append(data)
            labels_list.append(labels.cpu())
            
    return torch.cat(data_list), torch.cat(labels_list)

def getModelStdOutput(modelc : Model, dt_info):
    print(f"\nSaving std output for {modelc.name}")
    
    stdOutput = getStdOutputTensors(modelc, modelc.val_loader)
    
    torch.save(stdOutput, getSavePath(modelc, dt_info, False))    
    
def getModelEmbeddings(modelc : Model, dt_info):
    
    print(f"\nSaving {modelc.name} embeddings")
    
    data = getFeatureTensors(modelc.val_loader, modelc)
    
    torch.save(data, getSavePath(modelc, dt_info, True))
    
def writeModelOutputLine(modelc : Model, dt_info):
    std_output_path = getSavePath(modelc, dt_info, False)
    embedding_path = getSavePath(modelc, dt_info, True)
    
    data = [modelc.name, modelc.weights, modelc.source, dt_info.name_w_subset, std_output_path, embedding_path]
    
    writeCsvLine(modelOutputCsv_path, data)
    
def gatherAdditionalData(modelc : Model, dt_info : DtInfo, has_embedding=False):
    
    print(f"\nGathering additional data for {modelc.name}")
    
    data = [modelc.name, modelc.weights, modelc.source, dt_info.name_w_subset]
    
    ans = findInCsv(modelOutputCsv_path, ['model_name','model_weights','model_source','dataset(subset)'], data)
    
    if len(ans) == 0:
        getModelStdOutput(modelc, dt_info)
        
        if not has_embedding:
            getModelEmbeddings(modelc, dt_info)
        
        writeModelOutputLine(modelc, dt_info)
