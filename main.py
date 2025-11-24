# main.py

# Assuming you fixed the relative import issue
# For simplicity, I'll use a direct import of the module
import os
import sys
# If 'src' is one level up, add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the module containing the functions
import src.featureExtraction as featureExtraction 

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import torch.optim as optim
import torch.nn as nn
import time
from tqdm.auto import tqdm
from src import *

# --- Configuration ---
output_dir = "dataStorage"
os.makedirs(output_dir, exist_ok=True) # Ensure dataStorage folder exists
# ---------------------

data_transforms = {
    'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
}

if __name__ == "__main__":
    # --- Data Setup ---
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    class_names = train_dataset.classes
    
    # --- Device Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice set to: {device}")

    # --- Model Setup (Feature Extractors) ---
    
    # 1. ResNet-18 Feature Extractor
    first_model_name = 'resnet18'
    # Use num_classes=0 to get the feature vector *before* the classification head
    first_feature_extractor = timm.create_model(first_model_name, pretrained=True, num_classes=0).to(device)

    # 2. ConvNeXt-Tiny Feature Extractor
    second_model_name = 'convnext_tiny'
    second_feature_extractor = timm.create_model(second_model_name, pretrained=True, num_classes=0).to(device)
    
    # --- Feature Extraction (The Correct Step) ---
    
    print("\n--- Extracting Features for ResNet-18 ---")
    # This function iterates over all batches in train_loader and returns ONE large tensor
    first_features, _ = featureExtraction.extract_features_to_tensors(train_loader, first_feature_extractor)
    
    print("\n--- Extracting Features for ConvNeXt-Tiny ---")
    second_features, _ = featureExtraction.extract_features_to_tensors(train_loader, second_feature_extractor)

    # --- Saving the Full Embeddings ---
    
    first_output_path = os.path.join(output_dir, "first_global_embedding.pt")
    second_output_path = os.path.join(output_dir, "second_global_embedding.pt")

    torch.save(first_features, first_output_path)
    torch.save(second_features, second_output_path)

    print(f"\nSaved first embedding tensor (Shape: {first_features.shape}) to: {first_output_path}")
    print(f"Saved second embedding tensor (Shape: {second_features.shape}) to: {second_output_path}")

    print("\n--- Generating Text Representations ---")
    # Call the plotting function on the saved .pt file
    plot.plot_pt_file(first_output_path)
    plot.plot_pt_file(second_output_path)