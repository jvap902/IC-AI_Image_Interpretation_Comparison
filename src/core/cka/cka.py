import torch
from src.codifications import *
from src import config
from src.fileManagement.jsonUtils import getJsonInfo, updateJson

def ckaMethod(dt_info, fst_modelc, snd_modelc):
    
    paths = getJsonInfo(json_path=config.json_info_path, fields=["ckaData", "fst_embedding_path", "snd_embedding_path"])
    cka_results_folder = paths[0]+f"/{dt_info.name_w_subset}"
    fst_emb_path, snd_emb_path = paths[1], paths[2]
    
    m1_name, m2_name = modelCod(fst_modelc.source, fst_modelc.name, fst_modelc.weights), modelCod(snd_modelc.source, snd_modelc.name, snd_modelc.weights)
    
    fst_emb = torch.load(fst_emb_path, weights_only=True, map_location=torch.device('cpu'))
    snd_emb = torch.load(snd_emb_path, weights_only=True, map_location=torch.device('cpu'))
    
    cka_score = linearCKA(fst_emb, snd_emb).item()
    
    json_path = f"{cka_results_folder}/results.json"
    
    updateJson(json_path=json_path, fields=[f"{m1_name} {m2_name}"], values=[cka_score])

def linearCKA(X, Y):
    
    # centralização
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
    
    # numerador: ||Y.T @ X||_F^2
    numerator = torch.norm(torch.matmul(Y.t(), X), p='fro')**2
    
    # denominador: ||X.T @ X||_F * ||Y.T @ Y||_F
    denominator_x = torch.norm(torch.matmul(X.t(), X), p='fro')
    denominator_y = torch.norm(torch.matmul(Y.t(), Y), p='fro')
    
    result = numerator / (denominator_x * denominator_y)
    
    return result.detach().cpu().numpy()