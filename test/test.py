import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import torch.optim as optim
import torch.nn as nn
import time
from tqdm.auto import tqdm
from ..src import plot

# ... (Definitions of data_transforms, train_model, etc.) ...

# Define the transformation pipeline for CIFAR-10
data_transforms = {
    'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
}


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, freeze_backbone=False):
    """A simple function to train and validate a PyTorch model."""
    start_time = time.time()
    
    # Freeze backbone layers if specified
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name: # 'fc' is the default name for the classifier in resnet
                param.requires_grad = False
        print("Backbone frozen. Only training the classifier head.")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("Full model fine-tuning. All layers are trainable.")

    # Re-initialize optimizer to only include trainable parameters
    # The fix is to make sure this setup runs ONLY once in the main process
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-4) 

    # history must be initialized as a dictionary of lists/arrays, not a set of strings
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []} 

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=f"{phase.capitalize()} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())


    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return model, history



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
    num_epochs = 1 #só pra ver o plot
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, freeze_backbone=False)

    # Plot the results from our training run
    plot.plot_history(history)