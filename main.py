import argparse
import os
import sys
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
from scipy.stats import spearmanr, pearsonr

# If 'src' is one level up, add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Configuration ---
output_dir = "dataStorage"
os.makedirs(output_dir, exist_ok=True) # Ensure dataStorage folder exists

cache_dir = "datasetCache"
os.makedirs(cache_dir, exist_ok=True)
# ---------------------


data_transforms = {
    'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
}

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", type=int, required=False, help="Specify number of images per class in the dataset")
parser.add_argument("-m1", "--model1", type=str, required=False, help="Specify number of images per class in the dataset")
parser.add_argument("-m2", "--model2", type=str, required=False, help="Specify number of images per class in the dataset")
parser.add_argument("-d", "--dataset", type=str, required=False, help="Specify the dataset (cifar10 or cifar100)")

args = parser.parse_args()

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice set to: {device}")

    # --- Model Creation ---

    first_model_name = args.model1 if args.model1 else 'resnet50.a1_in1k'
    # Use num_classes=0 to get the feature vector *before* the classification head
    fst_model = timm.create_model(first_model_name, pretrained=True, num_classes=0).to(device)

    second_model_name = args.model2 if args.model2 else 'efficientnet_b0.ra_in1k'
    snd_model = timm.create_model(second_model_name, pretrained=True, num_classes=0).to(device)

    fst_model.eval()
    snd_model.eval()
    
    # --- Data Setup ---
    
    dataset_name = args.dataset if args.dataset else "cifar10"
    
    imagesPerClass = args.size if args.size else 100

    print(f"Number of images per class: {imagesPerClass}")

    fst_data_config = timm.data.resolve_model_data_config(fst_model)
    snd_data_config = timm.data.resolve_model_data_config(snd_model)

    fst_transforms = timm.data.create_transform(**fst_data_config, is_training=False) #assumindo que data_config do primeiro e segundo são iguais

    dataset = {}

    dataset['subset'], dataset['full_train'], dataset['val'] = loadDataset.getOrCreateDataset(
        data_dir='./data', 
        imagesPerClass=imagesPerClass, 
        transform=fst_transforms,
        cache_dir=cache_dir, # Use dataStorage for cache files
        dataset_name=dataset_name
    )
    
    batch_size = 64
    full_train_loader = DataLoader(dataset['full_train'], batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(dataset['subset'], batch_size=batch_size, shuffle=False, num_workers=4)
    val_dataset = DataLoader(dataset['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    
    class_names = dataset['full_train'].classes
    num_classes = len(class_names)

    # --- teste se modelos estão funcionando de acordo ---

    fst_acc = featureExtraction.train_and_validate_head(first_model_name, train_loader, val_dataset, epochs=10, num_classes=num_classes) #precisa dar uma leve treinada na nova cabeça para conseguir uma boa medida de accuracy
    snd_acc = featureExtraction.train_and_validate_head(second_model_name, train_loader, val_dataset, epochs=10, num_classes=num_classes) #precisa dar uma leve treinada na nova cabeça para conseguir uma boa medida de accuracy

    print(f"\n{first_model_name} Validation Accuracy: {fst_acc:.4f}")
    print(f"\n{second_model_name} Validation Accuracy: {snd_acc:.4f}")
    
    # --- Feature Extraction ---

    with torch.no_grad():
        print(f"\n--- Extracting Features for {first_model_name} ---")
        # This function iterates over all batches in train_loader and returns ONE large tensor
        first_features, _ = featureExtraction.extract_features_to_tensors(train_loader, fst_model)
        
        print(f"\n--- Extracting Features for {second_model_name} ---")
        second_features, _ = featureExtraction.extract_features_to_tensors(train_loader, snd_model)

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

    # --- Montando matriz ---

    fst_similarity_array = similarityAnalysis.cosineSimilarity(output_dir+"/first_global_embedding.pt")
    snd_similarity_array = similarityAnalysis.cosineSimilarity(output_dir+"/second_global_embedding.pt")

    #plot.similarityCsv(fst_similarity_array, output_dir+'/fst_similarity_array.csv', 1000, first_model_name)

    spearman, p_value = spearmanr(fst_similarity_array, snd_similarity_array)

    print(f"Spearman's Rank Correlation Coefficient (ρ): {spearman:.4f}")
    print(f"P-value: {p_value:.4e}")

    pearson, p_value = pearsonr(fst_similarity_array, snd_similarity_array)

    print(f"Pearson's Rank Correlation Coefficient (ρ): {pearson:.4f}")
    print(f"P-value: {p_value:.4e}")

    runData = [str(imagesPerClass), first_model_name, second_model_name, str(fst_acc), str(snd_acc), str(spearman), str(pearson), dataset_name]

    plot.collectRunData(output_dir+"/runData.csv", runData)