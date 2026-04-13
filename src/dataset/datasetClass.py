from ..csvUtils import *
from .loadDataset import *

class DtInfo:
    
    def __init__(self, dataset_name, subset, num_classes, num_images, same_classes, specific_classes):
        self.num_classes, self.num_images, self.name, self.subset = num_classes, num_images, dataset_name, subset
        self.images_per_class = self.num_images // self.num_classes
        self.name_w_subset = self.name.replace('/','-') + f"({self.subset})"
        
        self.classes = {}
                                
        if same_classes == None and not specific_classes:
            self.classes['train'], self.classes['validation'] = ['all'], ['all']
        elif specific_classes:
            self.classes['train'], self.classes['validation'] = self.getSpecificClasses()
        else:
            self.classes['train'], self.classes['validation'] = self.getSameClasses(same_classes)
            
            
    def getSpecificClasses(self):
        ans = findInCsv('./classes.csv', [], [])
        
        if len(ans) == 0:
            raise ValueError("Empty classes file")
        
        train_classes, validation_classes = getStringStrArray(ans[0]['train_classes']), getStringStrArray(ans[0]['validation_classes'])
        
        return train_classes, validation_classes
        
                                                
    def getSameClasses(self, same_classes, dataStorage_dir='./dataStorage'):
        from ..model.modelClass import Model
                
        base_dataset = DtInfo(same_classes[0], int(same_classes[1]), int(same_classes[2]), int(same_classes[3]), None)
        
        ans = findInCsv('./dataStorage/datasetClasses.csv', ['dataset', 'subset', 'num_classes', 'num_images'], [base_dataset.name, base_dataset.subset, base_dataset.num_classes, base_dataset.num_images])
        
        if len(ans) > 0:
            train_classes, validation_classes = getStringStrArray(ans[0]['train_classes']), getStringStrArray(ans[0]['validation_classes'])

        else:
            
            dt_name = base_dataset.name.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
            file_name = f"{dt_name}_subset_i{base_dataset.num_images}_c{base_dataset.num_classes}({base_dataset.subset}).pt"
            indices_file = os.path.join(dataStorage_dir, 'selectedIndices.csv')
            
            indices = csvUtils.findInCsv(indices_file, ['file_name'], [file_name])
            
            if (len(indices) == 0):
                raise ValueError("Dataset a ter classes copiadas não existe")
            
            dummy_model = Model('resnet18', 'torchvision', 'IMAGENET1K_V1')
            
            train_indices = csvUtils.getStringIntArray(indices[0]['train_indices'])
            val_indices = csvUtils.getStringIntArray(indices[0]['validation_indices'])
                        
            _, _ = loadIndicesFromDataset(base_dataset, train_indices, val_indices, './data', dummy_model)
            
            del dummy_model
            
            ans = findInCsv('./dataStorage/datasetClasses.csv', ['dataset', 'subset', 'num_classes', 'num_images'], [base_dataset.name, base_dataset.subset, base_dataset.num_classes, base_dataset.num_images])
        
            train_classes, validation_classes = getStringStrArray(ans[0]['train_classes']), getStringStrArray(ans[0]['validation_classes'])
        
        #writeCsvLine('./dataStorage/datasetClasses.csv', [self.name, self.subset, self.num_classes, self.num_images, train_classes, validation_classes])
        
        return train_classes, validation_classes