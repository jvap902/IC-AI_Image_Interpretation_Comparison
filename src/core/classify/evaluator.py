import torch
import torch.nn as nn
from typing import Tuple
from tqdm.auto import tqdm
import torch.optim as optim
from types import SimpleNamespace
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from src.config import device
from src.core.model import Model

# --- Utility to get features/labels from a DataLoader ---
def getFeatureTensors(modelc: Model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to extract all data (features or raw images) and labels 
    from a DataLoader into a single Tensor.
    
    If model is provided, it extracts features; otherwise, it extracts raw inputs.
    """
    data_list = []
    labels_list = []
    
    modelc.model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Extracting Data"):
            inputs = inputs.to(device)
            data = modelc.getEmbeddings(inputs)
                
            data_list.append(data)
            labels_list.append(labels.cpu())
            
    return torch.cat(data_list), torch.cat(labels_list)

def compute_metrics(pred):
    logits = torch.Tensor(pred.predictions)
    labels = pred.label_ids.astype(int)
    
    preds = logits.argmax(dim=1).numpy

    f1_mi = f1_score(labels, preds, average='micro')
    f1_ma = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1-macro': f1_ma,
        'f1-micro': f1_mi
    }

def evaluate(model, loader : DataLoader) -> dict:
    """
    Evaluates a model's accuracy on a given DataLoader.
    
    Args:
        test_loader: DataLoader for the validation set (features or images).
        model: The PyTorch model to evaluate.
        
    Returns:
        The accuracy (float) on the test set.
    """
    model.eval()
    
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluating Model"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    pred = SimpleNamespace(
        predictions=torch.cat(all_logits).numpy(),
        label_ids=torch.cat(all_labels).numpy()
    )

    return compute_metrics(pred)
    
def evaluateHead(modelc, embeddings):
    feature_tensor = TensorDataset(embeddings)
    feature_loader = DataLoader(feature_tensor)

    eval_dict = evaluate(modelc.head, feature_loader)

    return eval_dict