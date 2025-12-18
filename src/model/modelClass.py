from .modelCreation import getModel
from .classUtils import getExtractor

class Model:
    
    def __init__(self, model_name, model_source):
        self.name = model_name
        self.source = model_source
        
        self.model, self.data_transforms = getModel(self.source, self.name)
        
        self.featureExtractor = getExtractor(self.source)
        
    def extract(self, inputs):
        return self.featureExtractor(self, inputs)
        