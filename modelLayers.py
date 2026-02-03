import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import maxvit_t, resnet18, regnet_y_32gf, vit_b_16, efficientnet_b0, swin_t, convnext_tiny
from transformers import AutoModel, AutoImageProcessor
import clip
import open_clip
from src.model.modelCreation import get_default_openclip_pretrained

device = "cuda" if torch.cuda.is_available() else "cpu"

def torchvisionChildren(model_name):
    m = getattr(models, model_name)
    
    weights_enum = models.get_model_weights(m)
    
    weights_obj = weights_enum.DEFAULT

    model = m(weights=weights_obj).to(device)
    
    model.eval()
    
    return model.named_children

def clipChildren(model_name):
    model, preprocess = clip.load(model_name, device=device)
    
    model.eval()
    
    return model.named_children

def openClipChildren(model_name):
    pretrained = get_default_openclip_pretrained(model_name)
    
    model, _, data_transforms = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    
    return model.named_children

def dinoChildren(model_name):
    model = AutoModel.from_pretrained(
        model_name, 
        device_map="auto", 
    )
    return model.named_children

def getChildren(instance):
    if 'clip' == instance[0]:
        return clipChildren(instance[1])
    elif 'open_clip' == instance[0]:
        return openClipChildren(instance[1])
    elif 'dino' in instance[1]:
        return dinoChildren(instance[1])
    elif 'torchvision' == instance[0]:
        return torchvisionChildren(instance[1])

instances = [
    ('clip', 'ViT-B/32', 'DEFAULT'),
    ('clip', 'ViT-B/16', 'DEFAULT'),
    ('clip', 'ViT-L/14', 'DEFAULT'),
    ('open_clip', 'ViT-B-32-256', 'DEFAULT'),
    ('open_clip', 'ViT-B-16', 'DEFAULT'),
    ('open_clip', 'ViT-L-14', 'DEFAULT'),
    ('huggingface', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'DEFAULT'),
    ('huggingface', 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'DEFAULT'),
    ('torchvision', 'resnet18', 'IMAGENET1K_V1'),
    ('torchvision', 'resnet50', 'IMAGENET1K_V1'),
    ('torchvision', 'resnet152', 'IMAGENET1K_V1'),
    ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V1'),
    ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V2'),
    ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_SWAG_E2E_V1'),
    ('torchvision', 'regnet_y_32gf', 'IMAGENET1K_V2'),
    ('torchvision', 'vit_b_16', 'IMAGENET1K_V1'),
    ('torchvision', 'vit_b_16', 'IMAGENET1K_SWAG_E2E_V1'),
    ('torchvision', 'vit_l_16', 'IMAGENET1K_V1'),
    ('torchvision', 'vit_h_14', 'IMAGENET1K_SWAG_E2E_V1'),
    ('torchvision', 'maxvit_t', 'IMAGENET1K_V1'),
    ('torchvision', 'convnext_tiny', 'IMAGENET1K_V1'),
    ('torchvision', 'convnext_base', 'IMAGENET1K_V1'),
    ('torchvision', 'swin_t', 'IMAGENET1K_V1'),
    ('torchvision', 'swin_v2_t', 'IMAGENET1K_V1'),
    ('torchvision', 'efficientnet_b0', 'IMAGENET1K_V1'),
    ('torchvision', 'efficientnet_b4', 'IMAGENET1K_V1'),
    ('torchvision', 'efficientnet_b7', 'IMAGENET1K_V1')
]

if __name__ == "__main__":
    path = 'modelLayers.txt'
    
    children = clipChildren('ViT-L/14')
    
    with open(path, 'a') as file:
        for instance in instances:
            children = getChildren(instance)
            
            file.write(f"\n --- Model: {instance[1]}, source: {instance[0]} --- \n")

            file.write(f"{children} \n")