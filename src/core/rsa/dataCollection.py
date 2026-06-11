import clip
import torch
import open_clip
from tqdm.auto import tqdm
from src.core.model import Model
from src.core.dataset import DtInfo
from src.fileManagement.fileSystem import modelOutputSavePath
from src.fileManagement.csvUtils import findInCsv, writeCsvLine

modelOutputCsv_path = './dataStorage/model_output/modelOutput.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"

def logitExtractor(modelc : Model, dt_info, inputs):
    match modelc.source:
        case 'open_clip':
            tokenizer = open_clip.get_tokenizer(modelc.name)
            text = tokenizer(dt_info.val_classes).to(device)
            
            logits = modelc.model(inputs, text)
            
            return torch.tensor(logits[0]).clone().detach()
        
        case 'clip':
            text = clip.tokenize(dt_info.val_classes).to(device)
            logits_img, logits_txt = modelc.model(inputs, text)
            
            return torch.tensor(logits_img).clone().detach()
        
        case 'huggingface':
            if 'dinov3' in modelc.name:
                pixel_values = inputs['pixel_values']
        
                pixel_values = torch.stack(pixel_values)
                
                if pixel_values.dim() == 5 and pixel_values.size(0) == 1:
                    pixel_values = pixel_values.squeeze(0)
                
                pixel_values = pixel_values.to(device)

                with torch.inference_mode():
                    outputs = modelc.model(pixel_values=pixel_values)
                
                return torch.tensor(outputs[-1]).clone().detach()
            else:
                raise ValueError("Unsupported model")
            
        case _:
            return modelc.model(inputs)

def getStdOutputTensors(modelc : Model, loader, dt_info):
    data_list = []
    
    modelc.model.eval()
    
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Extracting Data"):
            inputs = inputs.to(device, non_blocking=True)
            modelc.model.eval()
            data = logitExtractor(modelc, dt_info, inputs)
            
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
        stdOutput = getStdOutputTensors(modelc, modelc.val_loader, dt_info)
        
        torch.save(stdOutput, modelOutputSavePath(modelc, dt_info, False)) 
        
        writeModelOutputLine(modelc, dt_info)
