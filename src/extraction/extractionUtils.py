import torch
from tqdm.auto import tqdm
from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

def extractFeatures(model, inputs, model_type):
    if (model_type == 'clip'):
        data = model.encode_image(inputs)
    else:
        data = model(inputs).cpu()
        
    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)
        
    return data.float()
        
# --- Utility to get features/labels from a DataLoader ---
def _extract_all_data(dataloader, model_type, model=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to extract all data (features or raw images) and labels 
    from a DataLoader into a single Tensor.
    
    If model is provided, it extracts features; otherwise, it extracts raw inputs.
    """
    data_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting Data"):
            inputs = inputs.to(device)
            if model:
                model.eval()
                data = extractFeatures(model, inputs, model_type) # Extract features using the backbone
            else:
                data = inputs.cpu() # Extract raw inputs
                
            data_list.append(data)
            labels_list.append(labels.cpu())
            
    return torch.cat(data_list), torch.cat(labels_list)