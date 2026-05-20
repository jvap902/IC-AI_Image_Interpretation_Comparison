import clip
import torch
import open_clip
from tqdm.auto import tqdm
from src.core.model import Model
from src.core.dataset import DtInfo
from src.fileManagement.fileSystem import embeddingSavePath
from src.fileManagement.csvUtils import findInCsv, writeCsvLine
from src.core.extraction.featureExtraction import getFeatureTensors

modelOutputCsv_path = './dataStorage/model_output/modelOutput.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"

def stdOutputExtractor(modelc : Model, dt_info : DtInfo, inputs):
    match modelc.source:
        case 'open_clip':
            tokenizer = open_clip.get_tokenizer(modelc.name)
            text = tokenizer(dt_info.val_classes).to(device)
            
            logits = modelc.model(inputs, text)
            
            return torch.tensor(logits[0])
        
        case 'clip':
            text = clip.tokenize(dt_info.val_classes).to(device)
            logits_img, logits_txt = modelc.model(inputs, text)
            
            logits_img = torch.tensor(logits_img)
            
            return logits_img
        
        case 'huggingface':
            if 'dinov3' in modelc.name:
                if isinstance(inputs, dict) or (hasattr(inputs, 'data') and 'pixel_values' in inputs):
                    # Already processed by a DataLoader/Processor
                    # Extract pixel_values and ensure they are on the correct device
                    pixel_values = inputs['pixel_values']
                    if isinstance(pixel_values, torch.Tensor):
                        inp = {'pixel_values': pixel_values.to(device)}
                    else:
                        # If it's a list or numpy array inside the dict, convert to tensor
                        inp = modelc.data_transforms(images=pixel_values, return_tensors="pt").to(device)
                else:
                    # Inputs are raw PIL images or Tensors that haven't been through the processor yet
                    inp = modelc.data_transforms(images=inputs, return_tensors="pt").to(device)

                with torch.inference_mode():
                    outputs = modelc.model(**inp)
                
                data = torch.tensor(outputs[-1])
                
                return data
            else:
                raise ValueError("Unsupported model")
            
        case _:
            return modelc.model(inputs)

def stdOutput(modelc : Model, dt_info : DtInfo, inputs):
    
    data = stdOutputExtractor(modelc, dt_info, inputs)
    
    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)
    
    return data.float()

def getStdOutputTensors(modelc : Model, dt_info : DtInfo, loader):
    data_list = []
    labels_list = []
    
    modelc.model.eval()
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Extracting Data"):
            inputs = inputs.to(device, non_blocking=True)
            if modelc.model:
                modelc.model.eval()
                data = stdOutput(modelc, dt_info, inputs) # Extract features using the backbone
            else:
                data = inputs.cpu() # Extract raw inputs
                            
            data = data.detach().cpu()
            labels = labels.detach().cpu()

            data_list.append(data)
            labels_list.append(labels)
            
    return torch.cat(data_list), torch.cat(labels_list)

def getModelStdOutput(modelc : Model, dt_info):
    print(f"\nSaving std output for {modelc.name}")
    
    stdOutput = getStdOutputTensors(modelc, dt_info, modelc.val_loader)
    
    torch.save(stdOutput, embeddingSavePath(modelc, dt_info, False))    
    
    del stdOutput
    
def getModelEmbeddings(modelc : Model, dt_info):
    
    print(f"\nSaving {modelc.name} embeddings")
    
    data = getFeatureTensors(modelc.val_loader, modelc)
    
    torch.save(data, embeddingSavePath(modelc, dt_info, True))
    
    del data
    
def writeModelOutputLine(modelc : Model, dt_info):
    std_output_path = embeddingSavePath(modelc, dt_info, False)
    embedding_path = embeddingSavePath(modelc, dt_info, True)
    
    data = [modelc.name, modelc.weights, modelc.source, dt_info.name_w_subset, std_output_path, embedding_path]
    
    writeCsvLine(modelOutputCsv_path, data)
    
def gatherAdditionalData(modelc : Model, dt_info : DtInfo, has_embedding=False):
    
    print(f"\nGathering additional data for {modelc.name}")
    
    data = [modelc.name, modelc.weights, modelc.source, dt_info.name_w_subset]
    
    ans = findInCsv(modelOutputCsv_path, ['model','model_weights','model_source','dataset(subset)'], data)
    
    if len(ans) == 0:
        getModelStdOutput(modelc, dt_info)
        
        if not has_embedding:
            getModelEmbeddings(modelc, dt_info)
        
        writeModelOutputLine(modelc, dt_info)
