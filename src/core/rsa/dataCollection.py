import clip
import torch
import open_clip
from tqdm.auto import tqdm
from src.core.model import Model
from src.core.dataset import DtInfo
from src.fileManagement.fileSystem import modelOutputSavePath
from src.fileManagement.csvUtils import findInCsv, writeCsvLine
from src.core.classify.evaluator import getFeatureTensors

modelOutputCsv_path = './dataStorage/model_output/modelOutput.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"

def getStdOutputTensors(modelc : Model, loader):
    data_list = []
    
    modelc.model.eval()
    
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Extracting Data"):
            inputs = inputs.to(device, non_blocking=True)
            modelc.model.eval()
            data = modelc.model(inputs)
            
            if data.dim() > 2:
                data = torch.flatten(data, start_dim=1)
            
            data = data.detach().cpu()

            data_list.append(data)
            
    return torch.cat(data_list)
    
def writeModelOutputLine(modelc : Model, dt_info):
    std_output_path = modelOutputSavePath(modelc, dt_info, False)
    embedding_path = modelOutputSavePath(modelc, dt_info, True)

    data = [modelc.name, modelc.weights, modelc.source, dt_info.name_w_subset, std_output_path, embedding_path]

    writeCsvLine(modelOutputCsv_path, data)
    
def gatherAdditionalData(modelc : Model, dt_info : DtInfo):
    
    print(f"\nGathering additional data")
    
    fst_data = [modelc.name, modelc.weights, modelc.source, dt_info.name_w_subset]
    
    ans = findInCsv(modelOutputCsv_path, ['model','model_weights','model_source','dataset(subset)'], fst_data)
    
    if len(ans) == 0:
        stdOutput = getStdOutputTensors(modelc, modelc.val_loader)
        
        torch.save(stdOutput, modelOutputSavePath(modelc, dt_info, False)) 
        
        writeModelOutputLine(modelc, dt_info)
