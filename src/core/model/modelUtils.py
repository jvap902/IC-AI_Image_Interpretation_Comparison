import clip
import torch
import open_clip
from huggingface_hub import login
from transformers import AutoModel, AutoImageProcessor
from torchvision.models import get_model, get_model_weights
from torchvision.models.feature_extraction import create_feature_extractor
from src.config import device
from src.core.dataset import datasetUtils

# model loading

def loadTorchvisionModel(modelc):
    model = get_model(modelc.name, weights=modelc.weights)

    weights_enum = get_model_weights(modelc.name)
    
    weights_obj = getattr(weights_enum, modelc.weights)
    
    preprocess = weights_obj.transforms()
    
    model.to(device)

    return model, preprocess

def loadClipModel(modelc):
    model, preprocess = clip.load(modelc.name, device=device)
    
    return model, preprocess

def loadHuggingfaceModel(modelc):
    
    hf_token = datasetUtils.loadToken('token.txt')
        
    login(token=hf_token, add_to_git_credential=False)
    
    model = AutoModel.from_pretrained(modelc.name, device_map="auto")
    preprocess = AutoImageProcessor.from_pretrained(modelc.name)
    
    return model, preprocess

def loadOpenClipModel(modelc):
    
    model, _, preprocess = open_clip.create_model_and_transforms(modelc.name, pretrained='laion2b_s34b_b79k') #talvez aqui fosse utilizado outro peso antes
    model.to(device)
    
    return model, preprocess

# Embedding extractors

def clipEmbeddingExtractor(modelc, inputs):
    with torch.no_grad():
        data = modelc.model.encode_image(inputs)
    
    return data

def generalEmbeddingExtractor(modelc, inputs):
    
    return_nodes = getReturnNodes(modelc.name)

    extractor = create_feature_extractor(modelc.model, return_nodes=return_nodes)
    
    output = extractor(inputs)
    
    data = output['global_embedding']
    
    return data

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

def dinov3EmbeddingExtractor(modelc, inputs):
    pixel_values = inputs['pixel_values']
            
    pixel_values = torch.stack(pixel_values)
    
    if pixel_values.dim() == 5 and pixel_values.size(0) == 1:
        pixel_values = pixel_values.squeeze(0)
    
    pixel_values = pixel_values.to(device)
    
    with torch.inference_mode():
        
        outputs = modelc.model(pixel_values=pixel_values)
    
        data = outputs.pooler_output
        
    return data