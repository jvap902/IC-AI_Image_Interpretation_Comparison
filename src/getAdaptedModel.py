import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

def getAdaptedModel():
    # Define the transformation pipeline for CIFAR-10
    # 1. Resize the images to the size expected by the pre-trained model (e.g., 224x224)
    # 2. Convert the images to PyTorch tensors
    # 3. Normalize the images with ImageNet's mean and standard deviation
    data_transforms = {
        'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    }

    # Download and load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=data_transforms['train'])
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=data_transforms['val'])

    # Create DataLoaders to feed data to the model in batches
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define class names
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Select a model architecture, e.g., resnet18 for a good balance of speed and performance
    model_name = 'resnet18'

    # Create the model
    # pretrained=True loads weights from ImageNet
    # num_classes=10 adapts the model for our CIFAR-10 task
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))

    # Let's inspect the final classifier layer to confirm the change
    print("Original ResNet-18 classifier:")
    print(timm.create_model(model_name, pretrained=True).get_classifier())

    print("\nOur modified classifier:")
    print(model.get_classifier())

    # Set up device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"\nModel moved to: {device}")

    return model

def loadTimmModels(first_model_name, second_model_name):
    # Use num_classes=0 to get the feature vector *before* the classification head
    fst_model = timm.create_model(first_model_name, pretrained=True, num_classes=0).to(device)
    snd_model = timm.create_model(second_model_name, pretrained=True, num_classes=0).to(device)

    fst_model.eval()
    snd_model.eval()
    
    fst_data_config = timm.data.resolve_model_data_config(fst_model)
    snd_data_config = timm.data.resolve_model_data_config(snd_model)
    
    data_transforms = timm.data.create_transform(**fst_data_config, is_training=False) #assumindo que data_config do primeiro e segundo são iguais

    return fst_model, snd_model, data_transforms

def loadTorchvisionModels(first_model_name, second_model_name):
    raise NotImplementedError