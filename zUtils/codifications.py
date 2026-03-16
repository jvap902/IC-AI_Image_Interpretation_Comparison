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
    'regnet_y_16gf': 13,
    'regnet_y_16gf': 14,
    'regnet_y_32gf': 15,
    'vit_b_16': 16,
    'vit_b_16': 17,
    'vit_l_16': 18,
    'vit_h_14': 19,
    'maxvit_t': 20,
    'convnext_tiny': 21,
    'convnext_base': 22,
    'swin_t': 23,
    'swin_v2_t': 24,
    'efficientnet_b0': 25,
    'efficientnet_b4': 26,
    'efficientnet_b7': 27
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
    model_letter = getModelNumber()
    train_letter = getTrainLetter()
    
    if 'clip' in src.lower():
        t = train_letter['CLIP']
    elif 'dinov3' in model:
        t = train_letter['DINOV3']
    else:
        t = train_letter[train]
        
    return f"{model_letter[model]}, {t}"