from .modelCreation import getModel
from .classUtils import getExtractor, get_attention_layers, get_conv_layers, is_vit_patch_embedding
import os
from src.dataset import loadDataset
from src import plot
from torch.utils.data import DataLoader

class Model:
    
    def __init__(self, model_name, model_source, weights="DEFAULT"):
        self.name = model_name
        self.source = model_source
        self.weights = weights
        
        self.model, self.data_transforms = getModel(self.source, self.name, weights)
        
        self.featureExtractor = getExtractor(self)
        
        self.archType = self.getArchitectureType()
            
    def extract(self, inputs):
        return self.featureExtractor(self, inputs)
    
    def getDataset(self, total_images, num_classes, dataset, subset_num=0, output_dir="./dataStorage", data_dir="./data"): #por enquanto apenas carregando datasets do Huggingface
        
        dt_name = dataset.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
        file_name = f"{dt_name}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
        indices_file = os.path.join(output_dir, 'selectedIndices.csv')
        
        if os.path.exists(indices_file):
            
            indices = plot.findInCsv(indices_file, ['file_name'], [file_name])
                
            if len(indices) != 0:
                train_indices = plot.getStringIntArray(indices[0]['train_indices'])
                val_indices = plot.getStringIntArray(indices[0]['validation_indices'])
                
                self.train_dataset, self.val_dataset = loadDataset.loadIndicesFromDataset(dataset, train_indices, val_indices, data_dir, self)
                
            else:
                self.train_dataset, self.val_dataset = loadDataset.createNewDataset(dataset, total_images, num_classes, output_dir, subset_num, data_dir, self)
                
        
        else:
            raise FileNotFoundError("Arquivo com indices não encontrado")
        
    
    def getLoaders(self, batch_size):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    def getArchitectureType(self): #ideia bem inicial, provalvemente será necessário um método mais complicado para isso
        
        if self.source == "timm":
            return self.model.default_cfg.get("architecture", "Unknown")
        
        conv_layers = get_conv_layers(self.model)
        att_layers  = get_attention_layers(self.model)

        has_conv = len(conv_layers) > 0
        has_att  = len(att_layers) > 0

        # Count ViT-style patch embeddings
        vit_patch_convs = [c for c in conv_layers if is_vit_patch_embedding(c)]

        # CNN only
        if has_conv and not has_att:
            return "CNN"

        # Transformer only
        if has_att:
            # Pure ViT: attention + exactly one patch conv
            if len(vit_patch_convs) == 1 and len(conv_layers) == 1:
                return "Vision Transformer"

            # Hybrid: CNN backbone + attention
            if len(conv_layers) > 1:
                return "Hybrid (CNN + Transformer)"

            return "Vision Transformer"

        return "Unknown"
        
        