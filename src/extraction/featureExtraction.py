import timm 
import torch
from tqdm.auto import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from typing import Tuple
from .extractionUtils import _extract_all_data

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_features_to_tensors(dataloader, model, model_type) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts features from a dataloader using a feature extractor model and returns them as tensors."""
    # Reuses the generalized internal utility
    return _extract_all_data(dataloader, model_type, model)


def evaluate_model(test_loader: DataLoader, model: nn.Module) -> float:
    """
    Evaluates a model's accuracy on a given DataLoader.
    
    Args:
        test_loader: DataLoader for the validation set (features or images).
        model: The PyTorch model to evaluate.
        
    Returns:
        The accuracy (float) on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating Model"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            #_, predicted = torch.max(outputs.data, 1)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    return accuracy

class LinearHead(nn.Module):
    """Simple linear classification head for the final features."""
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_and_validate_head_on_features(
    train_features_loader: DataLoader, 
    val_features_loader: DataLoader, 
    feature_dim: int, 
    epochs=10,
    num_classes=10
) -> float:
    """
    Trains a simple linear head directly on pre-extracted features (fast).
    
    Args:
        model_name: Name of the backbone model (for logging).
        train_features_loader: DataLoader containing the extracted feature tensors (training).
        val_features_loader: DataLoader containing the extracted feature tensors (validation).
        feature_dim: The dimensionality of the feature vectors (e.g., 2048 for ResNet50).
        epochs: Number of epochs to train the linear head.
        
    Returns:
        The final validation accuracy (float) after training the head.
    """
    print(f"\n--- Training Linear Head (on features, Epochs={epochs}) ---")
    
    # 1. Initialize the new Linear Head
    head_model = LinearHead(input_dim=feature_dim, num_classes=num_classes).to(device)
    
    # 2. Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head_model.parameters(), lr=0.01) # Increased learning rate is common for linear probes

    # 3. Training loop (Linear Probe)
    head_model.train()
    for epoch in tqdm(range(epochs), desc=f"Training Head"):
        for inputs, labels in train_features_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = head_model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

    # 4. Evaluation (on pre-extracted validation features)
    accuracy = evaluate_model(val_features_loader, head_model)
    print(f"Final Accuracy on Full Validation Set: {accuracy:.4f}")
    
    return accuracy


def train_and_validate_head(model: timm.create_model, train_loader: DataLoader, val_loader: DataLoader, epochs=15, num_classes=10, model_type='timm') -> float:
    """
    DEPRECATED: This function is now just a wrapper that performs feature extraction 
    and then calls the faster, feature-based training.
    """
    print(f"\n--- FAST VALIDATION WORKFLOW ---")
    
    # 1. Load model for feature extraction (without classification head)
    

    # 2. Extract features ONCE (the slow step, but only happens here)
    print("  Step 1/3: Extracting features from training subset...")
    train_features, train_labels = _extract_all_data(train_loader, model_type, model)
    
    print("  Step 2/3: Extracting features from full validation set...")
    val_features, val_labels = _extract_all_data(val_loader, model_type, model)

    feature_dim = train_features.size(1)

    # 3. Create DataLoaders for the extracted features (batching is still needed)
    batch_size = train_loader.batch_size # Re-use the batch size
    train_features_dataset = TensorDataset(train_features, train_labels)
    val_features_dataset = TensorDataset(val_features, val_labels)
    
    train_features_loader = DataLoader(train_features_dataset, batch_size=batch_size, shuffle=True)
    val_features_loader = DataLoader(val_features_dataset, batch_size=batch_size, shuffle=False)

    # 4. Train the head on features (the fast step)
    print("  Step 3/3: Training linear head on extracted features...")
    accuracy = train_and_validate_head_on_features(
        train_features_loader, 
        val_features_loader, 
        feature_dim, 
        epochs=epochs,
        num_classes=num_classes
    )

    return accuracy