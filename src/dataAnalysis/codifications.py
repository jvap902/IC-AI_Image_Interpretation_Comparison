def getModelNumber():
    model_number = {
    'facebook/dinov3-vitb16-pretrain-lvd1689m': 1,
    'facebook/dinov3-vitl16-pretrain-lvd1689m': 2,
    'ViT-B/32': 3,
    'ViT-B/16': 4,
    'ViT-L/14': 5,
    'ViT-B-32-256': 6,
    'ViT-B-16': 7,
    'ViT-L-14': 8,
    'resnet18': 9,
    'resnet50': 10,
    'resnet152': 11,
    'regnet_y_16gf': 12,
    'regnet_y_32gf': 13,
    'vit_b_16': 14,
    'vit_l_16': 15,
    'vit_h_14': 16,
    'maxvit_t': 17,
    'convnext_tiny': 18,
    'convnext_base': 19,
    'swin_t': 20,
    'swin_v2_t': 21,
    'efficientnet_b0': 22,
    'efficientnet_b4': 23,
    'efficientnet_b7': 24
    }
    
    return model_number

def getTrainLetter():
    train_letter = {
        "IMAGENET1K_V1": 'a',
        "IMAGENET1K_V2": 'b',
        "IMAGENET1K_SWAG_E2E_V1": 'c',
        "CLIP": 'd',
        "DINOV3": 'e'
    }
    
    return train_letter

def getModelTrainStr(src, model, train):
    model_number = getModelNumber()
    train_letter = getTrainLetter()
    
    if 'clip' in src.lower():
        t = train_letter['CLIP']
    elif 'dinov3' in model:
        t = train_letter['DINOV3']
    else:
        t = train_letter[train]
        
    return f"{model_number[model]}, {t}"

def getInstances():
    instances = [
        ('huggingface', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'DEFAULT'),
        ('huggingface', 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'DEFAULT'),
        ('clip', 'ViT-B/32', 'DEFAULT'),
        ('clip', 'ViT-B/16', 'DEFAULT'),
        ('clip', 'ViT-L/14', 'DEFAULT'),
        ('open_clip', 'ViT-B-32-256', 'DEFAULT'),
        ('open_clip', 'ViT-B-16', 'DEFAULT'),
        ('open_clip', 'ViT-L-14', 'DEFAULT'),
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

    return instances