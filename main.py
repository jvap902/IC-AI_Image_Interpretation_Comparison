import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from src import *
from src.model.modelClass import Model

# If 'src' is one level up, add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Configuration ---
output_dir = "dataStorage"
os.makedirs(output_dir, exist_ok=True) # Ensure dataStorage folder exists

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

cache_dir = "datasetCache"
os.makedirs(cache_dir, exist_ok=True)
# ---------------------

# data_transforms = {
#     'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
#     'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
# }

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", type=int, required=False, help="Specify number of images to be used from the dataset")
parser.add_argument("-m1", "--model1", type=str, required=False, help="Specify the model to be used as first model (within timm library)")
parser.add_argument("-m2", "--model2", type=str, required=False, help="Specify the model to be used as second model (within timm library)")
parser.add_argument("-d", "--dataset", type=str, required=False, help="Specify the dataset (cifar100, imagenet-a, imagenet-sketch or a link for huggingface dataset)")
parser.add_argument("-e", "--epochs", type=int, required=False, help="Specify the number of epochs to train the head for validation")
parser.add_argument("-nv", "--no_validation", action='store_false', help="Turns off model validation step")
parser.add_argument("--chunked", action='store_true', help="Enables spearman calculation in chunks to save memory")
parser.add_argument("--n_classes", type=int, required=False, help="Specify number of classes in the dataset (only for non cifar datasets)")
parser.add_argument("--specific_subset", type=int, required=False, help="Specify a specific subset number to load from cache")
parser.add_argument("--m1_source", type=str, required=False, default="timm", help="Specify from where the first model to be loaded come from, default is timm lib")
parser.add_argument("--m2_source", type=str, required=False, default="timm", help="Specify from where the second model to be loaded come from, default is timm lib")
parser.add_argument("--m1_weights", type=str, required=False, default="DEFAULT", help="Specify weights for torchvision models")
parser.add_argument("--m2_weights", type=str, required=False, default="DEFAULT", help="Specify weights for torchvision models")
parser.add_argument("-ed", "--existing_dissimilarity", action='store_true', required=False, default=False, help="Use previously calculated cossine dissimilarity for run")

args = parser.parse_args()

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice set to: {device}")

    # --- Model Creation ---

    first_model_name = args.model1 if args.model1 else 'resnet50.a1_in1k'
    second_model_name = args.model2 if args.model2 else 'efficientnet_b0.ra_in1k'
    
    fst_modelc = Model(first_model_name, args.m1_source, args.m1_weights)
    snd_modelc = Model(second_model_name, args.m2_source, args.m2_weights)
    
    #fst_model, data_transforms = modelCreation.getModel(args.m1_source, first_model_name)
    #snd_model, snd_dt = modelCreation.getModel(args.m2_source, second_model_name)
    
    # --- Data Setup ---
    
    dataset_name = args.dataset if args.dataset else "timm/mini-imagenet"
    
    total_images = args.size if args.size else 2000
    
    num_classes = args.n_classes if args.n_classes else 100
    
    if dataset_name == 'cifar10':
        num_classes = 10
    
    if total_images < num_classes:
        raise ValueError(f"Total images ({total_images}) must be at least equal to number of classes ({num_classes}).")
    
    epochs = args.epochs if args.epochs else 10

    print(f"\nNumber of images total: {total_images}")
    
    specific_subset = args.specific_subset if args.specific_subset is not None else 0

    fst_modelc.getDataset(total_images, num_classes, dataset_name, specific_subset, output_dir)
    snd_modelc.getDataset(total_images, num_classes, dataset_name, specific_subset, output_dir)
    
    batch_size = 64
    fst_modelc.getLoaders(batch_size)
    snd_modelc.getLoaders(batch_size)
    
    class_names = datasetUtils.get_classes(fst_modelc.train_dataset)
    num_classes = len(class_names)

    # --- teste se modelos estão funcionando de acordo ---
    fst_acc = 0.0
    snd_acc = 0.0
    if (args.no_validation):
        fst_acc = featureExtraction.train_and_validate_head(fst_modelc, epochs=epochs, num_classes=num_classes) #precisa dar uma leve treinada na nova cabeça para conseguir uma boa medida de accuracy
        snd_acc = featureExtraction.train_and_validate_head(snd_modelc, epochs=epochs, num_classes=num_classes) #precisa dar uma leve treinada na nova cabeça para conseguir uma boa medida de accuracy

        print(f"\n{first_model_name} Validation Accuracy: {fst_acc:.4f}")
        print(f"\n{second_model_name} Validation Accuracy: {snd_acc:.4f}")
    
    # --- Feature Extraction ---

    with torch.no_grad():
        print(f"\n--- Extracting Features for {first_model_name} ---")
        # This function iterates over all batches in val_loader and returns ONE large tensor
        first_features, _ = featureExtraction.getFeatureTensors(fst_modelc.val_loader, fst_modelc)
        
        print(f"\n--- Extracting Features for {second_model_name} ---")
        second_features, _ = featureExtraction.getFeatureTensors(snd_modelc.val_loader, snd_modelc)

    # --- Saving the Full Embeddings ---
    
    first_output_path = os.path.join(output_dir, "first_global_embedding.pt")
    second_output_path = os.path.join(output_dir, "second_global_embedding.pt")

    torch.save(first_features, first_output_path)
    torch.save(second_features, second_output_path)

    print(f"\nSaved first embedding tensor (Shape: {first_features.shape}) to: {first_output_path}")
    print(f"Saved second embedding tensor (Shape: {second_features.shape}) to: {second_output_path}")

    # print("\n--- Generating Text Representations ---")
    # # Call the plotting function on the saved .pt file
    # plot.plot_pt_file(first_output_path)
    # plot.plot_pt_file(second_output_path)

    # --- Montando matriz ---

    fst_dissimilarity_path = os.path.join(output_dir, "first_dissimilarity_array.pt")
    snd_dissimilarity_path = os.path.join(output_dir, "second_dissimilarity_array.pt")
    dissimilarity_csv_path = os.path.join(output_dir, "cosineDissimilarity.csv")
    np_folder = output_dir+"/dissimilarity_arrays"
    dt_name_w_subset = dataset_name+f"({specific_subset})"

    similarityAnalysis.getCosineDissimilarity(output_dir+"/first_global_embedding.pt", save_path=fst_dissimilarity_path, dissimilarity_csv=dissimilarity_csv_path, np_folder=np_folder, modelc=fst_modelc, dataset=dt_name_w_subset, previous_dissimilarity=args.existing_dissimilarity)
    
    similarityAnalysis.getCosineDissimilarity(output_dir+"/second_global_embedding.pt", save_path=snd_dissimilarity_path, dissimilarity_csv=dissimilarity_csv_path, np_folder=np_folder, modelc=snd_modelc, dataset=dt_name_w_subset, previous_dissimilarity=args.existing_dissimilarity)

    print("\nCalculating Pearson's correlation\n")
    pearson, p_value = similarityAnalysis.calculateCorrelations(fst_dissimilarity_path, snd_dissimilarity_path, correlation_type='pearson')

    print(f"Pearson's Rank Correlation Coefficient (ρ): {pearson:.4f}")

    spearman, p_value = similarityAnalysis.calculateCorrelations(fst_dissimilarity_path, snd_dissimilarity_path, correlation_type='spearman', chunked=args.chunked)

    print(f"Spearman's Rank Correlation Coefficient (ρ): {spearman:.4f}")

    runData = [str(total_images), str(num_classes), fst_modelc.source, fst_modelc.name, args.m1_weights, snd_modelc.source, snd_modelc.name, args.m2_weights, str(fst_acc), str(snd_acc), str(spearman), str(pearson), dt_name_w_subset]

    plot.writeCsvLine(output_dir+"/runData.csv", runData)