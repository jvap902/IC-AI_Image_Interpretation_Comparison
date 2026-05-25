import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.fileManagement import csvUtils, fileSystem
import os

device = config.device

def rdm(embedding, modelc, dt_info, csv_path):
    
    print(f"Calculating {modelc.name} RDM")
    
    embedding_np = embedding.detach().cpu().numpy()
    rdm = 1 - cosine_similarity(embedding_np)
    
    dissimilarity_path = fileSystem.dissimilaritySavePath(modelc, dt_info)
    
    np.save(dissimilarity_path, rdm)
    
    if len(csvUtils.findInCsv(csv_path, ['model', 'model_source', 'model_weights', 'dataset'], [modelc.name, modelc.source, modelc.weights, dt_info.name_w_subset])) == 0:
        csvUtils.writeCsvLine(csv_path, [modelc.name, modelc.source, modelc.weights, dt_info.name_w_subset, dissimilarity_path])
    
    return rdm

def rsm(fst_rdm, snd_rdm):
    idx = np.triu_indices(fst_rdm.shape[0], k=1)

    fst_rdm_flat = fst_rdm[idx]
    snd_rdm_flat = snd_rdm[idx]
    
    pearson_values = pearsonr(fst_rdm_flat, snd_rdm_flat)
    spearman_values = spearmanr(fst_rdm_flat, snd_rdm_flat)
    
    return pearson_values, spearman_values
    
def savedRdm(dt_name_w_subset, dissimilarity_csv, modelc):
    
    params = ['model', 'model_source', 'model_weights', 'dataset']
    values = [modelc.name, modelc.source, modelc.weights, dt_name_w_subset]
    
    ans = csvUtils.findInCsv(dissimilarity_csv, params, values)
    
    return ans