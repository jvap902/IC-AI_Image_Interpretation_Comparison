import argparse
import os
import sys
import torch
from src import *
from src.dataset.datasetClass import DtInfo
from src.model.modelClass import Model
import numpy as np

# If 'src' is one level up, add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", type=int, required=False, help="Specify number of images to be used from the dataset")
parser.add_argument("-m1", "--model1", type=str, required=False, help="Specify the model to be used as first model")
parser.add_argument("-m2", "--model2", type=str, required=False, help="Specify the model to be used as second model")
parser.add_argument("-d", "--dataset", type=str, required=False, help="Specify the dataset (cifar100, imagenet-a, imagenet-sketch, fgvc-aircraft or a link for huggingface dataset)")
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
parser.add_argument("-sc", "--same_classes", type=str, required=False, default=None, nargs='+', help="Specify [name, subset, num_classes, num_images] of a dataset for its classes to be used, only works for new subsets")
parser.add_argument("-out", "--output_file", type=str, required=False, default="./dataStorage/results/runData.csv", help="Specify path to file the run information will be written")

args = parser.parse_args()

fileSystem.makeFileSystem(args.output_file)

output_dir = 'dataStorage'


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice set to: {device}")

    # --- Model Creation ---

    first_model_name = args.model1 if args.model1 else 'resnet50.a1_in1k'
    second_model_name = args.model2 if args.model2 else 'efficientnet_b0.ra_in1k'
    
    fst_modelc = Model(first_model_name, args.m1_source, args.m1_weights)
    snd_modelc = Model(second_model_name, args.m2_source, args.m2_weights)
    
    # --- Data Setup ---
    
    dataset_name = args.dataset if args.dataset else "timm/mini-imagenet"
    
    total_images = args.size if args.size else 2000
    
    num_classes = args.n_classes if args.n_classes else 100
    
    specific_subset = args.specific_subset if args.specific_subset is not None else 0
    
    if dataset_name == 'cifar10':
        num_classes = 10
    
    if total_images < num_classes:
        raise ValueError(f"Total images ({total_images}) must be at least equal to number of classes ({num_classes}).")
    
    dt_info = DtInfo(dataset_name, specific_subset, num_classes, total_images, args.same_classes)
    
    epochs = args.epochs if args.epochs else 10

    print(f"\nNumber of images total: {total_images}")
    
    dissimilarity_csv_path = os.path.join(output_dir, "cosineDissimilarity.csv")
    dissimilarity_folder = output_dir+"/dissimilarity_arrays"

    fst_modelc.getDataset(dt_info, output_dir)
    snd_modelc.getDataset(dt_info, output_dir)
    
    batch_size = 64
    fst_modelc.getLoaders(batch_size)
    snd_modelc.getLoaders(batch_size)
    
    class_names = datasetUtils.getClasses(fst_modelc.train_dataset)
    num_classes = len(class_names)

    # --- teste se modelos estão funcionando de acordo ---
    fst_acc = 0.0
    snd_acc = 0.0
    
    validation_csv = output_dir+"/validation_results.csv"
    
    fst_val = plot.findInCsv(validation_csv, ["model","model_source","model_weights","dataset"], [fst_modelc.name, fst_modelc.source, fst_modelc.weights, dt_info.name_w_subset])
    
    snd_val = plot.findInCsv(validation_csv, ["model","model_source","model_weights","dataset"], [snd_modelc.name, snd_modelc.source, snd_modelc.weights, dt_info.name_w_subset])
    
    if (args.no_validation):
        
        if len(fst_val) == 0:
            fst_acc = featureExtraction.train_and_validate_head(fst_modelc, epochs=epochs, num_classes=num_classes) #precisa dar uma leve treinada na nova cabeça para conseguir uma boa medida de accuracy
            plot.writeCsvLine(validation_csv, [fst_modelc.name, fst_modelc.source, fst_modelc.weights, dt_info.name_w_subset, fst_acc])
        else:
            fst_acc = np.float32(fst_val[0]['accuracy'])
            
        if len(snd_val) == 0:
            snd_acc = featureExtraction.train_and_validate_head(snd_modelc, epochs=epochs, num_classes=num_classes) #precisa dar uma leve treinada na nova cabeça para conseguir uma boa medida de accuracy
            plot.writeCsvLine(validation_csv, [snd_modelc.name, snd_modelc.source, snd_modelc.weights, dt_info.name_w_subset, snd_acc])
        else:
            snd_acc = np.float32(snd_val[0]['accuracy'])

        print(f"\n{first_model_name} Validation Accuracy: {fst_acc:.4f}")
        print(f"\n{second_model_name} Validation Accuracy: {snd_acc:.4f}")
        
    
    get_fst_embedding = len(similarityAnalysis.isDissimilarityCalculated(dt_info.name_w_subset, dissimilarity_csv_path, fst_modelc)) == 0 or args.existing_dissimilarity == False
    get_snd_embedding = len(similarityAnalysis.isDissimilarityCalculated(dt_info.name_w_subset, dissimilarity_csv_path, snd_modelc)) == 0 or args.existing_dissimilarity == False    
    
    fst_embedding_path = dataCollection.getSavePath(fst_modelc, dt_info, True)
    snd_embedding_path = dataCollection.getSavePath(snd_modelc, dt_info, True)
    
    # --- Feature Extraction ---

    with torch.no_grad():
        
        if get_fst_embedding:            
            print(f"\n--- Extracting Features for {first_model_name} ---")
            # This function iterates over all batches in val_loader and returns ONE large tensor
            first_features, _ = featureExtraction.getFeatureTensors(fst_modelc.val_loader, fst_modelc)
            torch.save(first_features, fst_embedding_path)
            print(f"\nSaved first embedding tensor (Shape: {first_features.shape}) to: {fst_embedding_path}")
        
        if get_snd_embedding:
            print(f"\n--- Extracting Features for {second_model_name} ---")
            second_features, _ = featureExtraction.getFeatureTensors(snd_modelc.val_loader, snd_modelc)
            torch.save(second_features, snd_embedding_path)
            print(f"Saved second embedding tensor (Shape: {second_features.shape}) to: {snd_embedding_path}")

    # --- Montando matriz ---

    fst_dissimilarity_path = similarityAnalysis.getCosineDissimilarity(fst_embedding_path, dissimilarity_csv=dissimilarity_csv_path, dissimilarity_folder=dissimilarity_folder, modelc=fst_modelc, dt_info=dt_info, existing_dissimilarity=args.existing_dissimilarity)
    
    snd_dissimilarity_path = similarityAnalysis.getCosineDissimilarity(snd_embedding_path, dissimilarity_csv=dissimilarity_csv_path, dissimilarity_folder=dissimilarity_folder, modelc=snd_modelc, dt_info=dt_info, existing_dissimilarity=args.existing_dissimilarity)

    print("\nCalculating Pearson's correlation\n")
    pearson, p_value = similarityAnalysis.calculateCorrelations(fst_dissimilarity_path, snd_dissimilarity_path, correlation_type='pearson')

    print(f"Pearson's Rank Correlation Coefficient (ρ): {pearson:.4f}")

    spearman, p_value = similarityAnalysis.calculateCorrelations(fst_dissimilarity_path, snd_dissimilarity_path, correlation_type='spearman', chunked=args.chunked)

    print(f"Spearman's Rank Correlation Coefficient (ρ): {spearman:.4f}")

    runData = [str(total_images), str(num_classes), fst_modelc.source, fst_modelc.name, args.m1_weights, snd_modelc.source, snd_modelc.name, args.m2_weights, str(fst_acc), str(snd_acc), str(spearman), str(pearson), dt_info.name_w_subset]

    plot.writeCsvLine(args.output_file, runData)
    
    dataCollection.gatherAdditionalData(fst_modelc, dt_info, has_embedding=get_fst_embedding)
    dataCollection.gatherAdditionalData(snd_modelc, dt_info, has_embedding=get_snd_embedding)