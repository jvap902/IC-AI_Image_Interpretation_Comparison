import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64

json_info_path = "src/fileManagement/info.json"

cods = ["1e", "2e", "3d", "4d", "5d", "6d", "7d", "8d", "9a", "10a", "11a", "12a", "12b", "12c", "13b", "14a", "14c", "15a", "16c", "17a", "18a", "19a", "20a", "21a", "22a", "23a", "24a"]

datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0), ('imagenet-c-gaussian_noise-1', 0), ('imagenet-c-gaussian_noise-3', 0)]

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