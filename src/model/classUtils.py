from ..extraction.extractionUtils import clipExtractor, generalExtractor, huggingfaceExtractor, dinoExtractor
import torch.nn as nn

def getExtractor(modelc):
    
    model_type = modelc.source
    
    if (model_type == 'clip' or model_type == 'open_clip'):
        return clipExtractor
    elif ("dinov3" in modelc.name):
        return dinoExtractor
    elif (model_type == 'huggingface'):
        return huggingfaceExtractor
    else:
        return generalExtractor
    

def get_conv_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]

def get_attention_layers(model):
    attention_keywords = (
        "Attention",
        "MultiheadAttention",
        "Transformer",
        "SelfAttention",
        "MHSA",
    )

    return [m for m in model.modules() if any(k in m.__class__.__name__ for k in attention_keywords)]


def is_vit_patch_embedding(conv: nn.Conv2d):
    return (
        conv.in_channels == 3
        and conv.kernel_size == conv.stride
        and conv.kernel_size[0] >= 8
    )

def stripModelHead(modelc):
    """
    Identifies and replaces the classification head of a model with nn.Identity.
    Standardizes the backbone to return pooled feature maps.
    """
    
    match modelc.source:
        case 'torchvision':
            return stripTorchHead(modelc)
        case 'clip':
            modelc.model.ln_final = nn.Identity()
        case 'open_clip':
            modelc.model.ln_final = nn.Identity()
        case 'huggingface':
            if 'dinov3' in modelc.name:
                modelc.model.norm = nn.Identity()
            else:
                raise ValueError("Unsupported model")
        case _:
            raise NotImplementedError

    return modelc.model

def stripTorchHead(modelc):
    name = modelc.name.lower()
    model = modelc.model

    # EfficientNet (b0-b7)
    if "efficientnet" in name:
        model.classifier = nn.Sequential(nn.Identity())
        
    # ResNets and RegNets
    elif "resnet" in name or "regnet" in name:
        model.fc = nn.Identity()
        
    # MaxViT
    elif "maxvit" in name:        
        model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        
    # Vision Transformers (ViT)
    elif "vit" in name:
        model.heads = nn.Sequential(nn.Identity())
        
    # Swin Transformers
    elif "swin" in name:
        model.flatten = nn.Identity()
        model.head = nn.Identity() 
        
    # ConvNeXt
    elif "convnext" in name:
        model.classifier = nn.Sequential(nn.Identity())
        
    return model