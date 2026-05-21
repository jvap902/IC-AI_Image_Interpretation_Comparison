import torch
import torch.nn as nn
from typing import Tuple
from tqdm.auto import tqdm
import torch.optim as optim
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
    
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    f1_mi = f1_score(labels, preds, average='micro')
    f1_ma = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1-macro': f1_ma,
        'f1-micro': f1_mi
    }

def evaluate(modelc, val_features_loader : DataLoader) -> dict:
    """
    Evaluates a model's accuracy on a given DataLoader.
    
    Args:
        test_loader: DataLoader for the validation set (features or images).
        model: The PyTorch model to evaluate.
        
    Returns:
        The accuracy (float) on the test set.
    """
    
    modelc.head.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_features_loader, desc="Evaluating Model"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            #_, predicted = torch.max(outputs.data, 1)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    return accuracy


def extractFeatures(get_fst_embedding, get_snd_embedding, fst_embedding_path, snd_embedding_path, fst_modelc, snd_modelc):

    with torch.no_grad():
        
        if get_fst_embedding:            
            print(f"\n--- Extracting Features for {fst_modelc.name} ---")
            # This function iterates over all batches in val_loader and returns ONE large tensor
            first_features, _ = getFeatureTensors(fst_modelc.val_loader, fst_modelc)
            torch.save(first_features, fst_embedding_path)
            print(f"\nSaved first embedding tensor (Shape: {first_features.shape}) to: {fst_embedding_path}")
        
        if get_snd_embedding:
            print(f"\n--- Extracting Features for {snd_modelc.name} ---")
            second_features, _ = getFeatureTensors(snd_modelc.val_loader, snd_modelc)
            torch.save(second_features, snd_embedding_path)
            print(f"Saved second embedding tensor (Shape: {second_features.shape}) to: {snd_embedding_path}")
    