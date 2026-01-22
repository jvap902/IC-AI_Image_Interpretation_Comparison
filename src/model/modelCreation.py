import torch
import timm
import torchvision.models as models
import clip
import open_clip
from transformers import AutoModel, AutoImageProcessor
from huggingface_hub import login
from ..dataset import datasetUtils
from .classUtils import stripModelHead

device = "cuda" if torch.cuda.is_available() else "cpu"

def loadTimmModel(model_name):
    # Use num_classes=0 to get the feature vector *before* the classification head
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)

    model.eval()
    
    data_config = timm.data.resolve_model_data_config(model)
    
    data_transforms = timm.data.create_transform(**data_config, is_training=False) #assumindo que data_config do primeiro e segundo são iguais

    return model, data_transforms


def loadTorchvisionModel(model_name, weights="DEFAULT"):
    m = getattr(models, model_name)
    
    weights_enum = models.get_model_weights(m)
    
    if weights == "DEFAULT":
        weights_obj = weights_enum.DEFAULT
    
    elif isinstance(weights, str):
        try:
            weights_obj = getattr(weights_enum, weights)
        except AttributeError:
            raise ValueError(
                f"Invalid weights '{weights}' for model {model_name}. "
                f"Available: {[w.name for w in weights_enum]}")
    
    model = m(weights=weights_obj).to(device)
    
    #model = stripModelHead(model, model_name)
    
    model.eval()
    
    data_transforms = weights_obj.transforms() if weights_obj is not None else None   #assumindo que ambos modelos utilizam as mesmas transformações

    return model, data_transforms


def loadClipModel(model_name):
    model, data_transforms = clip.load(model_name, device=device)
    
    model.eval()
    
    return model, data_transforms


def loadHuggingfaceModel(model_name):
    
    hf_token = datasetUtils.loadToken('token.txt')
        
    login(token=hf_token, add_to_git_credential=False)
    
    model = AutoModel.from_pretrained(model_name, device_map="auto")
    data_transforms = AutoImageProcessor.from_pretrained(model_name)
    
    model.to(device)
    model.eval()
    
    return model, data_transforms

def loadOpenClipModel(model_name):
    
    pretrained = get_default_openclip_pretrained(model_name)
    
    model, _, data_transforms = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    
    return model, data_transforms
    
def get_default_openclip_pretrained(model_name: str) -> str:
    candidates = [
        p for m, p in open_clip.list_pretrained()
        if m == model_name
    ]

    if not candidates:
        raise ValueError(f"No pretrained weights found for {model_name}")

    # Preference order (best → fallback)
    priority = [
        "laion2b_s34b",
        "laion2b",
        "datacomp",
        "openai",
    ]

    for key in priority:
        for p in candidates:
            if key in p:
                return p

    # Absolute fallback
    return candidates[0]



def getModel(model_source, model_name, weights):
    
    print(f"\nLoading model {model_name}")
    
    match model_source:
        case 'timm':
            model, data_transforms = loadTimmModel(model_name)
            
        case 'torchvision':
            model, data_transforms = loadTorchvisionModel(model_name, weights=weights)
        
        case 'clip':
            model, data_transforms = loadClipModel(model_name)
            
        case 'huggingface':
            model, data_transforms = loadHuggingfaceModel(model_name)
            
        case 'open_clip':
            model, data_transforms = loadOpenClipModel(model_name)
            
        case _:
            raise ValueError(f"Not supported model source")
    
    return model, data_transforms