import timm
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader

# Global device definition is fine
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_all_logits(dataloader, model):
    """Gets all logits and labels from a dataloader."""
    all_logits = []
    all_labels = []
    
    # Ensure model is on the current device 
    model.to(device) 
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Getting Logits"):
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_logits), torch.cat(all_labels)


# The main execution logic must be wrapped
if __name__ == "__main__":

    # Define transforms
    data_transforms = {
        'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    }

    # === FIX: Model definitions MUST be inside the protected block ===
    # This prevents the worker processes from trying to initialize the models again.
    model_resnet = timm.create_model('resnet18', pretrained=True, num_classes=10).to(device)
    model_convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=10).to(device) 
    
    # Download and load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

    batch_size = 64

    # The DataLoaders are the source of the worker processes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Function calls are safe inside the block
    logits_resnet, labels_resnet = get_all_logits(val_loader, model_resnet)
    logits_convnext, labels_convnext = get_all_logits(val_loader, model_convnext)

    # Get the logit value of the highest-scoring class for each prediction
    confidences_resnet, _ = torch.max(logits_resnet, dim=1)
    confidences_convnext, _ = torch.max(logits_convnext, dim=1)

    # Plot the distribution of these top logit values
    plt.figure(figsize=(10, 6))
    sns.kdeplot(confidences_resnet.numpy(), label='ResNet18 Top Logit', fill=True)
    sns.kdeplot(confidences_convnext.numpy(), label='ConvNeXt-Tiny Top Logit', fill=True)
    plt.title('Distribution of Top Logit Values')
    plt.xlabel('Logit Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()