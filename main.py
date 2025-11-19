from src import *
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import torch.optim as optim
import torch.nn as nn
import time
from tqdm.auto import tqdm

data_transforms = {
    'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
}

if __name__ == "__main__":
    # Download and load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=data_transforms['train'])
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=data_transforms['val'])
    
    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Define class names
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Model Setup
    model_name = 'resnet18'
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    
    # Inspect classifier layers
    print("Original ResNet-18 classifier:")
    print(timm.create_model(model_name, pretrained=True).get_classifier())
    print("\nOur modified classifier:")
    print(model.get_classifier())
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"\nModel moved to: {device}")
    
    # --- Setup for training ---
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (note: it's re-initialized inside train_model, but we define it here for clarity)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # --- Execute Training ---
    num_epochs = 5
    model, history = extractionTraining.train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, freeze_backbone=False)