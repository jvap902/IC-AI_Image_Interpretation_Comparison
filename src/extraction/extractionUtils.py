import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

device = "cuda" if torch.cuda.is_available() else "cpu"

def clipExtractor(modelc, inputs):
    with torch.no_grad():
        data = modelc.model.encode_image(inputs)
    
    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)
        
    data = data / data.norm(dim=-1, keepdim=True)
    
    return data.float()

def generalExtractor(modelc, inputs):
    
    return_nodes = getReturnNodes(modelc.name)

    extractor = create_feature_extractor(modelc.model, return_nodes=return_nodes)
    
    output = extractor(inputs)
    
    # This is already pooled and normalized. No need for adaptive_avg_pool2d!
    data = output['global_embedding']
    
    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)
    
    data = data / data.norm(dim=-1, keepdim=True)
    
    return data.float()

def getReturnNodes(model_name):
    return_nodes = {}
    
    match model_name:
        case 'vit_b_16' | 'vit_l_16' | 'vit_h_14':
            return_nodes['getitem_5'] = 'global_embedding'
        case 'maxvit_t':
            return_nodes['classifier.0'] = 'global_embedding'
        case _:
            return_nodes['avgpool'] = 'global_embedding'
            
    return return_nodes

def huggingfaceExtractor(modelc, inputs):
    
    inp = modelc.data_transforms(images=inputs, return_tensors="pt").to(modelc.model.device)

    with torch.inference_mode():
        outputs = modelc.model(**inp)

    data = outputs.last_hidden_state  # (B, T, D)

    # pool tokens → feature vector
    data = data.mean(dim=1)  # (B, D)
    
    return data.float()

def dinoExtractor(modelc, inputs):
    if isinstance(inputs, dict) or hasattr(inputs, 'data'):
        pixel_values = inputs['pixel_values']
    elif isinstance(inputs, (list, tuple)):
        # Handles cases where the loader returns (image_tensor, label)
        pixel_values = inputs[0]
    else:
        # Standard tensor input
        pixel_values = inputs
            
    pixel_values = torch.stack(pixel_values)
    
    if pixel_values.dim() == 5 and pixel_values.size(0) == 1:
        pixel_values = pixel_values.squeeze(0)
    
    pixel_values = pixel_values.to(device)
    
    with torch.inference_mode():
        
        outputs = modelc.model(pixel_values=pixel_values)
    
        data = outputs.pooler_output

    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)

    return data.float()