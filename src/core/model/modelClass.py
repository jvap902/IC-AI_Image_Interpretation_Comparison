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
        return self.embeddingExtractor(self, inputs).float()
    
    def getLoaders(self, batch_size, train_dataset, val_dataset):
        
        train_preprocessed = self.preprocessDataset(train_dataset)
        val_preprocessed = self.preprocessDataset(val_dataset)
        
        self.train_loader = DataLoader(train_preprocessed, batch_size=batch_size, shuffle=False, num_workers=4)
        self.val_loader = DataLoader(val_preprocessed, batch_size=batch_size, shuffle=False, num_workers=4)
        
    def preprocessDataset(self, dataset):
        return PreprocessedDataset(dataset, self.preprocess)
    
    def setAcc(self, acc : float):
        self.acc = float(acc)
        
    def setHead(self, head):
        self.head = head
        
    def getEmbeddingExtractor(self):        
        if (self.source == 'clip' or self.source == 'open_clip'):
            return clipEmbeddingExtractor
        elif ('dinov3' in self.name.lower()):
            return dinov3EmbeddingExtractor
        else:
            return generalEmbeddingExtractor
        
class PreprocessedDataset(torch.utils.data.Dataset): #wrapper para que datasets de diferentes origens tenham o mesmo tipo/formato
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if isinstance(image, torch.Tensor) and image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        image = self.preprocess(image)
        if isinstance(image, torch.Tensor):
            image = image.clone()
        if isinstance(label, torch.Tensor):
            label = label.clone()
        return image, label