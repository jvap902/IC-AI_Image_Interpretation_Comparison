import timm 
import torch
from tqdm.auto import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utility to get features/labels from a DataLoader ---
def _extract_all_data(dataloader, model=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to extract all data (features or raw images) and labels 
    from a DataLoader into a single Tensor.
    
    If model is provided, it extracts features; otherwise, it extracts raw inputs.
    """
    data_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting Data"):
            inputs = inputs.to(device)
            if model:
                model.eval()
                data = model(inputs).cpu() # Extract features using the backbone
            else:
                data = inputs.cpu() # Extract raw inputs
                
            data_list.append(data)
            labels_list.append(labels.cpu())
            
    return torch.cat(data_list), torch.cat(labels_list)


def extract_features_to_tensors(dataloader, model) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts features from a dataloader using a feature extractor model and returns them as tensors."""
    # Reuses the generalized internal utility
    return _extract_all_data(dataloader, model)


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
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

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
    model_name: str, 
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
    print(f"\n--- Training Linear Head for {model_name} (on features, Epochs={epochs}) ---")
    
    # 1. Initialize the new Linear Head
    head_model = LinearHead(input_dim=feature_dim, num_classes=num_classes).to(device)
    
    # 2. Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head_model.parameters(), lr=0.01) # Increased learning rate is common for linear probes

    # 3. Training loop (Linear Probe)
    head_model.train()
    for epoch in tqdm(range(epochs), desc=f"Training Head ({model_name})"):
        for inputs, labels in train_features_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = head_model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

    # 4. Evaluation (on pre-extracted validation features)
    accuracy = evaluate_model(val_features_loader, head_model)
    print(f"  {model_name} Final Accuracy on Full Validation Set: {accuracy:.4f}")
    
    return accuracy


def train_and_validate_head(model_name: str, train_loader: DataLoader, val_loader: DataLoader, epochs=15, num_classes=10) -> float:
    """
    DEPRECATED: This function is now just a wrapper that performs feature extraction 
    and then calls the faster, feature-based training.
    """
    print(f"\n--- FAST VALIDATION WORKFLOW for {model_name} ---")
    
    # 1. Load model for feature extraction (without classification head)
    try:
        feature_extractor = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    except Exception as e:
        print(f"Error: Could not load model '{model_name}'. Exception: {e}")
        return 0.0

    # 2. Extract features ONCE (the slow step, but only happens here)
    print("  Step 1/3: Extracting features from training subset...")
    train_features, train_labels = _extract_all_data(train_loader, feature_extractor)
    
    print("  Step 2/3: Extracting features from full validation set...")
    val_features, val_labels = _extract_all_data(val_loader, feature_extractor)

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
        model_name, 
        train_features_loader, 
        val_features_loader, 
        feature_dim, 
        epochs=epochs,
        num_classes=num_classes
    )

    return accuracy