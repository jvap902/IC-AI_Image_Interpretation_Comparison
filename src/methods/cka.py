from torch_cka import CKA
import torch
from tqdm import tqdm
from warnings import warn
import os
from zUtils.codifications import *
from ..fileSystem.fileSystem import getJsonInfo

import time
import numpy as np

def cka(dt_info, fst_modelc, snd_modelc):
    
    output_dir = getJsonInfo(["output_dir"])[0]
    cka_feat_folder = output_dir+f"/ckaFeatures/{dt_info.name_w_subset}"
    
    m1_name, m2_name = getModelTrainStr(fst_modelc.source, fst_modelc.name, fst_modelc.weights), getModelTrainStr(snd_modelc.source, snd_modelc.name, snd_modelc.weights)
    
    cka = modifiedCka(fst_modelc.model, snd_modelc.model,
              model1_name=m1_name, model2_name=m2_name,
              #model1_layers=fst_modelc.model.named_modules, model2_layers=snd_modelc.model.named_modules, a princípio extrai de todos por padrão
              device='cuda' if torch.cuda.is_available() else 'cpu')
    
    cka.setNewAtt(m1_name, m2_name, cka_feat_folder)
    
    cka.compare(dataloader1=fst_modelc.val_loader, dataloader2=snd_modelc.val_loader)

    cka.plot_results()

    return


#modificando método de comparação para salvar em disco os resultados de um modelo para não ser necessário recalcular para a próxima comparação
class modifiedCka(CKA):
    
    def setNewAtt(self, m1_name, m2_name, output_folder):
        self.m1_name = m1_name
        self.m2_name = m2_name
        
        self.cka_folder = output_folder
        
        os.makedirs(self.cka_folder, exist_ok=True)
        
    def saveHsics(self):
        # Save the Model 1 self-similarities
        torch.save({
            'name': self.model1_name,
            'layers': self.model1_layers,
            'self_hsic': self.hsic_matrix[:, 0, 0].cpu()
        }, f"{self.model1_name}_cka_cache.pt")

        # Save the Model 2 self-similarities
        torch.save({
            'name': self.model2_name,
            'layers': self.model2_layers,
            'self_hsic': self.hsic_matrix[0, :, 2].cpu()
        }, f"{self.model2_name}_cka_cache.pt")
    
    def compare(self, dataloader1, dataloader2 = None):
        self.originalComparePart(dataloader1=dataloader1, dataloader2=dataloader2)

        num_batches = min(len(dataloader1), len(dataloader1))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):            
            
            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))
            
            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):                    
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches
                    

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())
        
        self.saveHsics(self.hsic_matrix)

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"
        
    
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