from typing import List,Tuple

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

def getNumberModel(number=None):
    number_model = {
    1: 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    2: 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    3: 'ViT-B/32',
    4: 'ViT-B/16',
    5: 'ViT-L/14',
    6: 'ViT-B-32-256',
    7: 'ViT-B-16',
    8: 'ViT-L-14',
    9: 'resnet18',
    10: 'resnet50',
    11: 'resnet152',
    12: 'regnet_y_16gf',
    13: 'regnet_y_32gf',
    14: 'vit_b_16',
    15: 'vit_l_16',
    16: 'vit_h_14',
    17: 'maxvit_t',
    18: 'convnext_tiny',
    19: 'convnext_base',
    20: 'swin_t',
    21: 'swin_v2_t',
    22: 'efficientnet_b0',
    23: 'efficientnet_b4',
    24: 'efficientnet_b7'
    }

    if number == None:
        return number_model
    else:
        return number_model[number]

def getTrainLetter():
    train_letter = {
        "IMAGENET1K_V1": 'a',
        "IMAGENET1K_V2": 'b',
        "IMAGENET1K_SWAG_E2E_V1": 'c',
        "CLIP": 'd',
        "DINOV3": 'e'
    }
    
    return train_letter

def getLetterTrain(letter=None):
    letter_train = {
        'a': "IMAGENET1K_V1",
        'b': "IMAGENET1K_V2",
        'c': "IMAGENET1K_SWAG_E2E_V1",
        'd': "CLIP",
        'e': "DINOV3"
    }

    if letter == None:
        return letter_train
    else:
        return letter_train[letter]

def dtNameSubset(dt: Tuple[str, int] | List[Tuple[str, int]]):
    if isinstance(dt, tuple):
        name, subset = dt
        return f"{name.replace('/', '-')}({subset})"
    
    # Otherwise, treat it as a list of tuples
    return [f"{d[0].replace('/', '-')}({d[1]})" for d in dt]

def getModelTrainStr(src, model, train):
    model_number = getModelNumber()
    train_letter = getTrainLetter()
    
    if 'clip' in src.lower():
        t = train_letter['CLIP']
    elif 'dinov3' in model:
        t = train_letter['DINOV3']
    else:
        t = train_letter[train]
        
    return f"{model_number[model]}{t}"

def codToInstace(number: int, letter: chr):
    instances = getInstances()
    
    model = getNumberModel(number)
    train = getLetterTrain(letter)
    
    if 'dinov3' in model:
        inst = ('huggingface', model,  'DEFAULT')
        return (instances.index(inst), inst)
    
    elif train == 'CLIP':
        if number < 6:
            inst = ('clip', model, 'DEFAULT')
        else:
            inst = ('open_clip', model, 'DEFAULT')
            
        return (instances.index(inst), inst)
    
    else:
        inst = ('torchvision', model, train)
    
    return (instances.index(inst), inst)

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

if __name__ == "__main__":
    instances = getInstances()

    for inst in instances:
        print(f"{getModelTrainStr(inst[0], inst[1], inst[2])} - {inst}")
        
#1e - ('huggingface', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'DEFAULT')
#2e - ('huggingface', 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'DEFAULT')
#3d - ('clip', 'ViT-B/32', 'DEFAULT')
#4d - ('clip', 'ViT-B/16', 'DEFAULT')
#5d - ('clip', 'ViT-L/14', 'DEFAULT')
#6d - ('open_clip', 'ViT-B-32-256', 'DEFAULT')
#7d - ('open_clip', 'ViT-B-16', 'DEFAULT')
#8d - ('open_clip', 'ViT-L-14', 'DEFAULT')
#9a - ('torchvision', 'resnet18', 'IMAGENET1K_V1')
#10a - ('torchvision', 'resnet50', 'IMAGENET1K_V1')
#11a - ('torchvision', 'resnet152', 'IMAGENET1K_V1')
#12a - ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V1')
#12b - ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V2')
#12c - ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_SWAG_E2E_V1')
#13b - ('torchvision', 'regnet_y_32gf', 'IMAGENET1K_V2')
#14a - ('torchvision', 'vit_b_16', 'IMAGENET1K_V1')
#14c - ('torchvision', 'vit_b_16', 'IMAGENET1K_SWAG_E2E_V1')
#15a - ('torchvision', 'vit_l_16', 'IMAGENET1K_V1')
#16c - ('torchvision', 'vit_h_14', 'IMAGENET1K_SWAG_E2E_V1')
#17a - ('torchvision', 'maxvit_t', 'IMAGENET1K_V1')
#18a - ('torchvision', 'convnext_tiny', 'IMAGENET1K_V1')
#19a - ('torchvision', 'convnext_base', 'IMAGENET1K_V1')
#20a - ('torchvision', 'swin_t', 'IMAGENET1K_V1')
#21a - ('torchvision', 'swin_v2_t', 'IMAGENET1K_V1')
#22a - ('torchvision', 'efficientnet_b0', 'IMAGENET1K_V1')
#23a - ('torchvision', 'efficientnet_b4', 'IMAGENET1K_V1')
#24a - ('torchvision', 'efficientnet_b7', 'IMAGENET1K_V1')