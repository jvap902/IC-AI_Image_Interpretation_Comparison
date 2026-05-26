import torch
import argparse
from pathlib import Path
from src import config
from .model import Model
from .dataset import DtInfo
from .cka.cka import ckaMethod
from .rsa.rsa import rsaMethod
from .classify import evaluator, newHead
from src.fileManagement import fileSystem, jsonUtils, csvUtils

available_methods = ["rsa", "cka"]

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", type=int, default=2000, required=False, help="Specify number of images to be used from the dataset")
parser.add_argument("-m1", "--model1", type=str, required=False, help="Specify the model to be used as first model")
parser.add_argument("-m2", "--model2", type=str, required=False, help="Specify the model to be used as second model")
parser.add_argument("-d", "--dataset", type=str, default="ILSVRC/imagenet-1k", required=False, help="Specify the dataset (cifar10, cifar100, imagenet-a, imagenet-sketch, fgvc-aircraft, imagenet-c-$distortion$-$level$ or a link for huggingface dataset)")
parser.add_argument("-e", "--epochs", type=int, default=10, required=False, help="Specify the number of epochs to train the head for validation")
parser.add_argument("-nv", "--no_validation", action='store_true', help="Turns off model validation step")
parser.add_argument("--n_classes", type=int, default=100, required=False, help="Specify number of classes in the dataset (only for non cifar datasets)")
parser.add_argument("-ss", "--specific_subset", default=0, type=int, required=False, help="Specify a specific subset number to load from cache")
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
    
    print(f"\nDevice set to: {config.device}")

    # --- Model Creation ---
    first_model_name = args.model1 if args.model1 else 'resnet18'
    second_model_name = args.model2 if args.model2 else 'resnet50'
    
    fst_modelc = Model(first_model_name, args.m1_source, args.m1_weights)
    snd_modelc = Model(second_model_name, args.m2_source, args.m2_weights)
    
    # --- Data Setup ---
    dataset_name = args.dataset
    total_images = args.size    
    num_classes = args.n_classes    
    specific_subset = args.specific_subset
    
    if dataset_name == 'cifar10':
        num_classes = 10
    
    if total_images < num_classes:
        raise ValueError(f"Total images ({total_images}) must be at least equal to number of classes ({num_classes}).")
    
    if (args.same_classes != None and args.new_dt_specific_classes):
        raise ValueError(f"Can't specify two different set of classes")
    
    dt_info = DtInfo(dataset_name, specific_subset, num_classes, total_images, args.same_classes, args.new_dt_specific_classes)
    
    epochs = args.epochs

    print(f"\nNumber of images total: {total_images}")

    train_dataset, val_dataset = dt_info.getDatasets()
    
    batch_size = config.batch_size
    fst_modelc.getLoaders(batch_size, train_dataset, val_dataset)
    snd_modelc.getLoaders(batch_size, train_dataset, val_dataset)
    
    class_names = dt_info.train_classes
    num_classes = len(class_names)
    

    # --- Embedding Extraction ---
    fst_embedding_path = fileSystem.modelOutputSavePath(fst_modelc, dt_info, embedding=True)
    snd_embedding_path = fileSystem.modelOutputSavePath(snd_modelc, dt_info, embedding=True)
    
    fields = [f"fst_embedding_path", "snd_embedding_path"]
    values = [fst_embedding_path, snd_embedding_path]
    jsonUtils.updateJson(config.json_info_path, fields, values)
        
    if not Path(fst_embedding_path).is_file() or args.reextract:
        print("Extracting first model embeddings\n")
        fst_emb = evaluator.getFeatureTensors(fst_modelc, fst_modelc.val_loader)
        torch.save(fst_emb, fst_embedding_path)
    else:
        print("Loading first model saved embeddings")
        fst_emb = torch.load(fst_embedding_path, weights_only=True)
    
    if not Path(snd_embedding_path).is_file() or args.reextract: 
        print("Extracting second model embeddings\n")
        snd_emb = evaluator.getFeatureTensors(snd_modelc, snd_modelc.val_loader)
        torch.save(snd_emb, snd_embedding_path)
    else:
        print("Loading second model saved embeddings\n")
        snd_emb = torch.load(snd_embedding_path, weights_only=True)

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
        val_labels_tensor = torch.Tensor([label for _, label in val_dataset])
        
        newHead.newTrainedHead(fst_modelc, dt_info.num_classes, epochs=epochs)
        fst_eval_dict = evaluator.evaluateHead(fst_modelc, fst_emb, val_labels_tensor)
        fst_modelc.setAcc(fst_eval_dict["accuracy"])
        csvUtils.writeCsvLine(validation_csv, [fst_modelc.name, fst_modelc.source, fst_modelc.weights, dt_info.name_w_subset, fst_modelc.acc])
            
        newHead.newTrainedHead(snd_modelc, dt_info.num_classes, epochs=epochs)
        snd_eval_dict = evaluator.evaluateHead(snd_modelc, snd_emb, val_labels_tensor)
        snd_modelc.setAcc(snd_eval_dict["accuracy"])
        csvUtils.writeCsvLine(validation_csv, [snd_modelc.name, snd_modelc.source, snd_modelc.weights, dt_info.name_w_subset, snd_modelc.acc])

    print(f"\n{first_model_name} Validation Accuracy: {fst_modelc.acc:.4f}")
    print(f"{second_model_name} Validation Accuracy: {snd_modelc.acc:.4f}\n")

    # --- Experiment execution ---
    match args.method:
        case "rsa":
            rsaMethod(dt_info, fst_modelc, fst_emb, snd_modelc, snd_emb, args)
        case "cka":
            ckaMethod(dt_info, fst_modelc, fst_emb, snd_modelc, snd_emb)
        case "validation":
            print("validado")
        case _:
            raise ValueError(f"Method {args.method} not recognized. The available methods are: {available_methods}")
        
if __name__ == "__main__":
    main()