from torch_cka import CKA
from functools import partial
import torch
from tqdm import tqdm
from warnings import warn
import os
from src.codifications import *
from ..fileSystem.fileSystem import getJsonInfo, updateJson, createFile
import numpy as np
import pandas as pd
from itertools import groupby

def cka(dt_info, fst_modelc, snd_modelc):
    
    output_dir = getJsonInfo(["output_dir"])[0]
    cka_folder = output_dir+f"/ckaData/{dt_info.name_w_subset}"
    
    m1_name, m2_name = getModelTrainStr(fst_modelc.source, fst_modelc.name, fst_modelc.weights), getModelTrainStr(snd_modelc.source, snd_modelc.name, snd_modelc.weights)
    
    cka = modifiedCka(fst_modelc.model, snd_modelc.model,
              model1_name=m1_name, model2_name=m2_name,
              model1_layers=getModelLayer(fst_modelc.name), model2_layers=getModelLayer(snd_modelc.name), #a princípio extrai de todos por padrão
              device='cuda' if torch.cuda.is_available() else 'cpu')
    
    cka.setNewAtt(m1_name, m2_name, cka_folder)
    
    cka.compare(dataloader1=fst_modelc.val_loader, dataloader2=snd_modelc.val_loader)

    #cka.plot_results(save_path=f"{cka_folder}/images/{cka.m1_name}_{cka.m2_name}.png")
    dic = cka.export()
        
    dic["CKA"] = dic['CKA'].cpu().numpy().tolist()
    
    del dic["dataset1_name"]
    del dic["dataset2_name"]
    
    json_path = f"{cka_folder}/results.json"
    
    updateJson([f"{cka.m1_name} {cka.m2_name}"], [dic], json_path=json_path)

    return

def getModelLayer(model_name):
    match model_name:
        case 'vit_b_16' | 'vit_l_16' | 'vit_h_14':
            return ['encoder.ln']
            #return ['getitem_5']
        case 'maxvit_t':
            return ['classifier.0']
        case 'facebook/dinov3-vitb16-pretrain-lvd1689m' | 'facebook/dinov3-vitl16-pretrain-lvd1689m':
            return ['embeddings']
        case 'ViT-B/32' | 'ViT-B/16' | 'ViT-L/14' | 'ViT-B-32-256' | 'ViT-B-16' | 'ViT-L-14':
            return['visual.ln_post']
        case _:
            return ['avgpool']


def jsonCkaToDataFrame(json_path):
    data = getJsonInfo(json_path=json_path)
    instances = getInstances()
    
    matrix = np.zeros((len(instances), len(instances)))
    
    labels = []
    for key, data_dic in data.items():
        
        models = key.split(' ')
        
        labels.append(data_dic["model1_name"])
        labels.append(data_dic["model2_name"])
        
        m1 = [models[0][:-1], models[0][-1]]
        m2 = [models[1][:-1], models[1][-1]]
        
        i, _ = codToInstace(m1[0], m1[1])
        j, _ = codToInstace(m2[0], m2[1])
        
        matrix[i,j] = data_dic["CKA"]
        matrix[j,i] = data_dic["CKA"]
        
    np.fill_diagonal(matrix, 1.0)
    
    df = pd.DataFrame(matrix)
    
    labels = [label for _, label in groupby(labels)]
    
    df.columns = labels
    df.index = labels
    
    return df

#modificando método de comparação para salvar em disco os resultados de um modelo para não ser necessário recalcular para a próxima comparação
class modifiedCka(CKA):
    
    def setNewAtt(self, m1_name, m2_name, output_folder):
        self.m1_name = m1_name
        self.m2_name = m2_name
        
        self.cka_folder = output_folder
        
        self.hsic1_path = f"{self.cka_folder}/cka_matrices/{self.m1_name}_cka.pt"
        self.hsic2_path = f"{self.cka_folder}/cka_matrices/{self.m2_name}_cka.pt"
        
        os.makedirs(self.cka_folder, exist_ok=True)
        os.makedirs(f"{self.cka_folder}/cka_matrices", exist_ok=True)
        createFile(f"{self.cka_folder}/results.json", "{}")
        os.makedirs(f"{self.cka_folder}/images", exist_ok=True)
        
        
    def saveHsics(self, m1: bool, m2: bool):
        if m1:
            # Save the Model 1 self-similarities
            torch.save(self.hsic_matrix[:, 0, 0].cpu(), self.hsic1_path)

        if m2:
            # Save the Model 2 self-similarities
            torch.save(self.hsic_matrix[0, :, 2].cpu(), self.hsic2_path)
    
    def compare(self, dataloader1, dataloader2 = None):
        
        self.originalComparePart(dataloader1=dataloader1, dataloader2=dataloader2)

        num_batches = min(len(dataloader1), len(dataloader1))

        self.getHsicMatrix(dataloader1, dataloader2, num_batches)

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"
    
    
    def getTensor(self, x):
        #para suportar modelos de fora do pytorch
        
        if hasattr(x, 'pixel_values'): #dinov3
            tensor = torch.stack(x['pixel_values']).to(self.device)
            if tensor.dim() == 5 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            
            return tensor
        
        else: #pytorch
            return x
    
    
    def originalComparePart(self, dataloader1, dataloader2 = None):
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)
        
        
    def getHsicMatrix(self, dataloader1, dataloader2, num_batches):
    
        hsic1 = os.path.isfile(self.hsic1_path)
        hsic2 = os.path.isfile(self.hsic2_path)
        
        if hsic1:
            cached_hsic1 = torch.load(self.hsic1_path, weights_only=True, map_location=torch.device('cpu'))
            self.hsic_matrix[:, :, 0] = cached_hsic1.view(-1, 1)
            print(f"\n --- Loaded cached HSIC for {self.m1_name} ---")
    
        if hsic2:
            cached_hsic2 = torch.load(self.hsic2_path, weights_only=True, map_location=torch.device('cpu'))
            self.hsic_matrix[:, :, 2] = cached_hsic2.view(1, -1)
            print(f"\n --- Loaded cached HSIC for {self.m2_name} ---")

        with torch.no_grad():
            for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):            
                
                self.model1_features = {}
                self.model2_features = {}
                m1_dtype = next(self.model1.parameters()).dtype
                m2_dtype = next(self.model2.parameters()).dtype
                
                tensor1 = self.getTensor(x1)
                tensor2 = self.getTensor(x2)
                
                if hasattr(self.model1, 'visual'): #para suportar CLIP
                    # Move tensor to device and pass to the visual backbone only
                    _ = self.model1.visual(tensor1.to(self.device).type(m1_dtype))
                else:
                    _ = self.model1(tensor1.to(self.device))

                if hasattr(self.model2, 'visual'):
                    # Move tensor to device and pass to the visual backbone only
                    _ = self.model2.visual(tensor2.to(self.device).type(m2_dtype))
                else:
                    _ = self.model2(tensor2.to(self.device))
                                
                for i, (name1, feat1) in enumerate(self.model1_features.items()):
                    X = feat1.flatten(1).to(torch.float32)
                    K = X @ X.t()
                    K.fill_diagonal_(0.0)
                    
                    if not hsic1: self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches
                    
                    for j, (name2, feat2) in enumerate(self.model2_features.items()):
                        Y = feat2.flatten(1).to(torch.float32)
                        L = Y @ Y.t()
                        L.fill_diagonal_(0)
                        assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
                        
                        if not hsic2: self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches
                        
                        self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                        
                        del Y, L
                    del X, K
                    
                self.model1_features.clear()
                self.model2_features.clear()
        
        
        self.saveHsics(not hsic1, not hsic2)