import torch
from src.codifications import modelCod
from src import config
from src.fileManagement.jsonUtils import getJsonInfo, updateJson

def ckaMethod(dt_info, fst_modelc, fst_emb, snd_modelc, snd_emb):
    
    cka_results_folder = getJsonInfo(json_path=config.json_info_path, fields=["ckaData"])[0]+f"/{dt_info.name_w_subset}"
    
    m1_name, m2_name = modelCod(fst_modelc.source, fst_modelc.name, fst_modelc.weights), modelCod(snd_modelc.source, snd_modelc.name, snd_modelc.weights)
    
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