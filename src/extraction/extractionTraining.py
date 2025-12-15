import torch.optim as optim
import torch.nn as nn
import time
from tqdm.auto import tqdm
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, freeze_backbone=False):
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
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-4) # Use a small learning rate for fine-tuning

    history = {'train_loss', 'val_loss', 'train_acc', 'val_acc'}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloader, desc=f"{phase.capitalize()} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return model, history


def extractionTraining(device, model, train_loader, val_loader):
    # --- Setup for training ---
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (will be re-initialized in the train function)
    # Note: For linear probing, a higher learning rate might be better (e.g., 1e-3)
    # For full fine-tuning, a lower learning rate is crucial (e.g., 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # --- Execute Training ---
    # Example: Run full fine-tuning for 5 epochs
    num_epochs = 5
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, freeze_backbone=False)
