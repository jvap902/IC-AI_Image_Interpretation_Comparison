import torch
from tqdm import tqdm
from warnings import warn
import os
from src.codifications import *
from src import config
from ..fileManagement.fileSystem import createFile
from ..fileManagement.jsonUtils import getJsonInfo, updateJson
import numpy as np
import pandas as pd
from itertools import groupby

def ckaMethod(dt_info, fst_modelc, snd_modelc):
    
    paths = getJsonInfo(json_path=config.json_info_path, fields=["ckaData", "fst_embedding_path", "snd_embedding_path"])
    cka_results_folder = paths[0]+f"/{dt_info.name_w_subset}"
    fst_emb_path, snd_emb_path = paths[1], paths[2]
    
    m1_name, m2_name = getModelTrainStr(fst_modelc.source, fst_modelc.name, fst_modelc.weights), getModelTrainStr(snd_modelc.source, snd_modelc.name, snd_modelc.weights)
    
    fst_emb = torch.load(fst_emb_path)
    snd_emb = torch.load(snd_emb_path)
    
    cka_score = cka(fst_emb, snd_emb)
    
    json_path = f"{cka_results_folder}/results.json"
    
    updateJson(json_path=json_path, fields=[f"{m1_name} {m2_name}"], values=[cka_score])

    del dic


def cka(X, Y):
    
    # numerador: ||Y.T @ X||_F^2
    numerator = torch.norm(torch.matmul(Y.t(), X), p='fro')**2
    
    # denominador: ||X.T @ X||_F * ||Y.T @ Y||_F
    denominator_x = torch.norm(torch.matmul(X.t(), X), p='fro')
    denominator_y = torch.norm(torch.matmul(Y.t(), Y), p='fro')
    
    return numerator / (denominator_x * denominator_y)
    
    

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
    instances = config.instances
    
    matrix = np.zeros((len(instances), len(instances)))
    
    labels = []
    for key, data_dic in data.items():
        
        models = key.split(' ')
        
        labels.append(data_dic["model1_name"])
        labels.append(data_dic["model2_name"])
        
        m1 = [models[0][:-1], models[0][-1]]
        m2 = [models[1][:-1], models[1][-1]]
        
        i, _ = codToInstance(m1[0], m1[1])
        j, _ = codToInstance(m2[0], m2[1])
        
        matrix[i,j] = data_dic["CKA"]
        matrix[j,i] = data_dic["CKA"]
        
    np.fill_diagonal(matrix, 1.0)
    
    df = pd.DataFrame(matrix)
    
    labels = [label for _, label in groupby(labels)]
    
    df.columns = labels
    df.index = labels
    
    return df