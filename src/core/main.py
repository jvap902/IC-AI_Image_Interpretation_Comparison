import os
import sys
import torch
import argparse
from src import config
from .model import Model
from .dataset import DtInfo
from .cka.cka import ckaMethod
from .rsa.rsa import rsaMethod
from .classify import evaluator, newHead
from src.fileManagement import fileSystem, jsonUtils, csvUtils

# If 'src' is one level up, add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

available_methods = ["rsa", "cka"]

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", type=int, required=False, help="Specify number of images to be used from the dataset")
parser.add_argument("-m1", "--model1", type=str, required=False, help="Specify the model to be used as first model")
parser.add_argument("-m2", "--model2", type=str, required=False, help="Specify the model to be used as second model")
parser.add_argument("-d", "--dataset", type=str, required=False, help="Specify the dataset (cifar10, cifar100, imagenet-a, imagenet-sketch, fgvc-aircraft, imagenet-c-$distortion$-$level$ or a link for huggingface dataset)")
parser.add_argument("-e", "--epochs", type=int, required=False, help="Specify the number of epochs to train the head for validation")
parser.add_argument("-nv", "--no_validation", action='store_true', help="Turns off model validation step")
parser.add_argument("--chunked", action='store_true', help="Enables spearman calculation in chunks to save memory")
parser.add_argument("--n_classes", type=int, required=False, help="Specify number of classes in the dataset (only for non cifar datasets)")
parser.add_argument("-ss", "--specific_subset", type=int, required=False, help="Specify a specific subset number to load from cache")
parser.add_argument("--m1_source", type=str, required=False, default="torchvision", help="Specify from where the first model to be loaded come from, default is pytorch")
parser.add_argument("--m2_source", type=str, required=False, default="torchvision", help="Specify from where the second model to be loaded come from, default is pytorch")
parser.add_argument("-m1_w", "--m1_weights", type=str, required=False, default="IMAGENET1K_V1", help="Specify weights for torchvision models")
parser.add_argument("-m2_w", "--m2_weights", type=str, required=False, default="IMAGENET1K_V1", help="Specify weights for torchvision models")
parser.add_argument("-ed", "--existing_dissimilarity", action='store_true', required=False, default=False, help="Use previously calculated cossine dissimilarity for run")
parser.add_argument("-sc", "--same_classes", type=str, required=False, default=None, nargs='+', help="Specify [name, subset, num_classes, num_images] of a dataset for its classes to be used, only works for new subsets")
parser.add_argument("-out", "--output_file", type=str, required=False, default="./dataStorage/rsaData/runData.csv", help="Specify path to file the run information will be written")
parser.add_argument("--reextract", action='store_true', required=False, default=False, help="Force embedding extraction even when there are saved embeddings")
parser.add_argument("-ndsc", "--new_dt_specific_classes", action='store_true', required=False, default=False, help="Specify arbitrary classes in a classes.csv file, only works for new subsets")
parser.add_argument("-met", "--method", type=str, required=False, default="rsa", help=f"Specify the method to be used, the available methods are: {available_methods}")

args = parser.parse_args()

def main():
    
    fileSystem.makeFileSystem(args.output_file)
    paths = jsonUtils.getJsonInfo(config.json_info_path)
    
    device = config.device
    
    output_dir = paths["output_dir"]
    
    print(f"\nDevice set to: {device}")

    # --- Model Creation ---
    first_model_name = args.model1 if args.model1 else 'resnet18'
    second_model_name = args.model2 if args.model2 else 'resnet50'
    
    fst_modelc = Model(first_model_name, args.m1_source, args.m1_weights)
    snd_modelc = Model(second_model_name, args.m2_source, args.m2_weights)
    
    # --- Data Setup ---
    dataset_name = args.dataset if args.dataset else "ILSVRC/imagenet-1k"
    
    total_images = args.size if args.size else 2000
    
    num_classes = args.n_classes if args.n_classes else 100
    
    specific_subset = args.specific_subset if args.specific_subset is not None else 0
    
    if dataset_name == 'cifar10':
        num_classes = 10
    
    if total_images < num_classes:
        raise ValueError(f"Total images ({total_images}) must be at least equal to number of classes ({num_classes}).")
    
    if (args.same_classes != None and args.new_dt_specific_classes):
        raise ValueError(f"Can't specify two different set of classes")
    
    dt_info = DtInfo(dataset_name, specific_subset, num_classes, total_images, args.same_classes, args.new_dt_specific_classes)
    
    epochs = args.epochs if args.epochs else 10

    print(f"\nNumber of images total: {total_images}")

    train_dataset, val_dataset = dt_info.getDatasets()
    
    batch_size = 64
    fst_modelc.getLoaders(batch_size, train_dataset, val_dataset)
    snd_modelc.getLoaders(batch_size, train_dataset, val_dataset)

    print(f"First loader: {type(fst_modelc.train_loader)}")
    print(f"\nSecond loader: {type(snd_modelc.train_loader)}")
    
    class_names = dt_info.train_classes
    num_classes = len(class_names)
    

    # --- Embedding Extraction ---
    fst_embedding_path, snd_embedding_path = fileSystem.modelOutputSavePath(fst_modelc, snd_modelc, dt_info, embedding=True)
    
    if args.reextract: 
        print("Extraindo embeddings\n")
        fst_embeddings, _ = evaluator.getFeatureTensors(fst_modelc, fst_modelc.val_loader)
        snd_embeddings, _ = evaluator.getFeatureTensors(snd_modelc, snd_modelc.val_loader)
        
        torch.save(fst_embeddings, fst_embedding_path)
        torch.save(snd_embeddings, snd_embedding_path)
    
    else:
        fst_embeddings = torch.load(fst_embedding_path, weights_only=True)
        snd_embeddings = torch.load(snd_embedding_path, weights_only=True)


    # --- Model validation ---
    validation_csv = paths["validation_results"]
    
    fst_val = csvUtils.findInCsv(validation_csv, ["model","model_source","model_weights","dataset"], [fst_modelc.name, fst_modelc.source, fst_modelc.weights, dt_info.name_w_subset])
    snd_val = csvUtils.findInCsv(validation_csv, ["model","model_source","model_weights","dataset"], [snd_modelc.name, snd_modelc.source, snd_modelc.weights, dt_info.name_w_subset])
    
    fst_acc = 0 if len(fst_val) == 0 else fst_val[0]['accuracy']
    snd_acc = 0 if len(snd_val) == 0 else snd_val[0]['accuracy']

    if args.no_validation:
        fst_modelc.setAcc(fst_acc)
        snd_modelc.setAcc(snd_acc)
    else:
        newHead.newTrainedHead(fst_modelc, dt_info.num_classes, epochs=10)
        fst_eval_dict = evaluator.evaluateHead(fst_modelc, fst_embeddings)
        fst_modelc.setAcc(fst_eval_dict["accuracy"])
        csvUtils.writeCsvLine(validation_csv, [fst_modelc.name, fst_modelc.source, fst_modelc.weights, dt_info.name_w_subset, fst_modelc.acc])
            
        newHead.newTrainedHead(snd_modelc, dt_info.num_classes, epochs=10)
        snd_eval_dict = evaluator.evaluateHead(snd_modelc, snd_embeddings)
        snd_modelc.setAcc(snd_eval_dict["accuracy"])
        csvUtils.writeCsvLine(validation_csv, [snd_modelc.name, snd_modelc.source, snd_modelc.weights, dt_info.name_w_subset, snd_modelc.acc])

        print(f"\n{first_model_name} Validation Accuracy: {fst_modelc.acc:.4f}")
        print(f"\n{second_model_name} Validation Accuracy: {snd_modelc.acc:.4f}")

    # --- Experiment execution ---
    match args.method:
        case "rsa":
            rsaMethod(dt_info, fst_modelc, snd_modelc, total_images, num_classes, args)
        case "cka":
            ckaMethod(dt_info, fst_modelc, snd_modelc)
        case "validation":
            print("validado")
        case _:
            raise ValueError(f"Method {args.method} not recognized. The available methods are: {available_methods}")
        
if __name__ == "__main__":
    main()