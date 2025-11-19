from src import *

import numpy as np
from tqdm.auto import tqdm
import timm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_features_to_tensors(dataloader, model):
    """Extracts features from a dataloader using a feature extractor model and returns them as tensors."""
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            features = model(inputs)
            
            features_list.append(features.cpu())
            labels_list.append(labels.cpu())
            
    # Concatenate all batches
    all_features = torch.cat(features_list)
    all_labels = torch.cat(labels_list)
    
    return all_features, all_labels

if __name__ == "__main__":
    data_transforms = {
        'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    }

    # Use the pooled feature extractor from Method 1
    feature_extractor = timm.create_model('resnet18', pretrained=True, num_classes=0).to(device)

    # Download and load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Extract features for train and validation sets
    train_features, train_labels = extract_features_to_tensors(train_loader, feature_extractor)
    val_features, val_labels = extract_features_to_tensors(val_loader, feature_extractor)

    print(f"Shape of training features tensor: {train_features.shape}")
    print(f"Shape of training labels tensor: {train_labels.shape}")


    # TRAINING AND EVALUATING THE LINEAR CLASSIFIER

   # 1. Create TensorDatasets and DataLoaders from the extracted features
    train_feat_dataset = TensorDataset(train_features, train_labels)
    val_feat_dataset = TensorDataset(val_features, val_labels)

    # Use a larger batch size since we are not loading/transforming images
    feature_batch_size = 256 
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=feature_batch_size, shuffle=True)
    val_feat_loader = DataLoader(val_feat_dataset, batch_size=feature_batch_size, shuffle=False)

    # 2. Define the simple linear classifier
    class LinearClassifier(nn.Module):
        def __init__(self, in_features, num_classes):
            super().__init__()
            self.linear = nn.Linear(in_features, num_classes)
        
        def forward(self, x):
            return self.linear(x)
        
    class_names = train_dataset.classes

    # 3. Instantiate the model, loss, and optimizer
    # Get the number of input features from the tensor shape
    num_in_features = train_features.shape[1]
    num_classes = len(class_names)

    linear_model = LinearClassifier(num_in_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_model.parameters(), lr=0.001)

    # 4. Training loop
    print("Training simple linear classifier on extracted features...")
    num_epochs = 20
    for epoch in range(num_epochs):
        linear_model.train()
        for features, labels in train_feat_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = linear_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        linear_model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for features, labels in val_feat_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = linear_model(features)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        
        accuracy = 100 * total_correct / total_samples
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')

    print(f"Final Linear Classifier Validation Accuracy: {accuracy:.2f}%")



