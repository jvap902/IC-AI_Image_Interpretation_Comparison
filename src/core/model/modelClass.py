import os
from torch.utils.data import DataLoader
from src.fileManagement import csvUtils
from core.dataset import loadDataset, DtInfo
from .modelCreation import getModel
from .classUtils import getExtractor, stripModelHead

class Model:
    
    def __init__(self, model_name, model_source, weights="DEFAULT"):
        self.name = model_name
        self.source = model_source
        self.weights = weights
        
        self.model, self.data_transforms = getModel(self.source, self.name, weights)
        
        self.model = stripModelHead(self)
        
        self.featureExtractor = getExtractor(self)
                    
    def extract(self, inputs):
        return self.featureExtractor(self, inputs)
    
    def getDataset(self, dt_info: DtInfo, output_dir="./dataStorage", data_dir="./data"):
        
        dt_name = dt_info.name.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
        file_name = f"{dt_name}_subset_i{dt_info.num_images}_c{dt_info.num_classes}({dt_info.subset}).pt"
        indices_file = os.path.join(output_dir, 'selectedIndices.csv')
        
        if os.path.exists(indices_file):
            
            indices = csvUtils.findInCsv(indices_file, ['file_name'], [file_name])
                
            if len(indices) != 0:
                train_indices = csvUtils.getStringIntArray(indices[0]['train_indices'])
                val_indices = csvUtils.getStringIntArray(indices[0]['validation_indices'])
                
                self.train_dataset, self.val_dataset = loadDataset.loadIndicesFromDataset(dt_info, train_indices, val_indices, data_dir, self)
                
            else:
                self.train_dataset, self.val_dataset = loadDataset.createNewDataset(dt_info, output_dir, data_dir, self)
            
            print(f"\nLoaded {len(dt_info.train_classes)} train classes and {len(dt_info.val_classes)} validation classes")
            print(f"Validation dataset has {len(self.val_dataset)} total images")
        
        else:
            raise FileNotFoundError("Arquivo com indices não encontrado")
        
    
    def getLoaders(self, batch_size):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    def setAcc(self, acc):
        self.acc = acc
        