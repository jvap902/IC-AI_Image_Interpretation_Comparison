import os
from torch.utils.data import DataLoader
from .modelUtils import *
from src.fileManagement import csvUtils

class Model:
    
    def __init__(self, model_name, model_source, weights="DEFAULT"):
        self.name = model_name
        self.source = model_source
        self.weights = weights
        
        self.getModel()
        self.model.eval()
        
        self.embeddingExtractor = self.getEmbeddingExtractor()
        
    def getModel(self):
        print(f"\nLoading model {self.name}")
    
        match self.source:                
            case 'torchvision':
                self.model, self.preprocess = loadTorchvisionModel(self)
            
            case 'clip':
                self.model, self.preprocess = loadClipModel(self)
                
            case 'huggingface':
                self.model, self.preprocess = loadHuggingfaceModel(self)
                
            case 'open_clip':
                self.model, self.preprocess = loadOpenClipModel(self)
                
            case _:
                raise ValueError(f"Not supported model source")
                    
    def getEmbeddings(self, inputs):
        return self.embeddingExtractor(self, inputs)
    
    def getLoaders(self, batch_size, train_dataset, val_dataset):
        
        #train_preprocessed = 
        #val_preprocessed = 
        
        self.train_loader = DataLoader(train_preprocessed, batch_size=batch_size, shuffle=False, num_workers=4)
        self.val_loader = DataLoader(val_preprocessed, batch_size=batch_size, shuffle=False, num_workers=4)
        
    def preprocessDataset(self, dataset):
        raise NotImplemented
    
    def setAcc(self, acc):
        self.acc = acc
        
    def getEmbeddingExtractor(self):        
        if (self.source == 'clip' or self.source == 'open_clip'):
            return clipEmbeddingExtractor
        elif (self.source == 'huggingface'):
            return huggingfaceEmbeddingExtractor
        else:
            return generalEmbeddingExtractor