import torch.nn.functional as F
import torch
import numpy as np
import math
from scipy.stats import pearsonr, spearmanr

from src.fileManagement import csvUtils
from . import similarityUtils
from src import memoryManagement
from ..model.modelClass import Model
from ..dataset.datasetClass import DtInfo
import os

# Constants for memory calculation (assuming torch.float32)
FLOAT_BYTES = 4 

device = "cuda" if torch.cuda.is_available() else "cpu"

total_memory_gb = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

def cosineDissimilarity(ptPath, csv_path, dissimilarity_path, modelc: Model, dt_name_w_subset): #retorna vetor de dissimilaridades de cosseno
    
    print(f"Loding feature tensor for cosine dissimilarity \n")
    
    feature_tensor = torch.load(ptPath, map_location='cpu', weights_only=True).to(device)
    numEmbeddings = feature_tensor.size(0)

    print(f"Feature tensor loaded with shape: {feature_tensor.shape}\n")
    print(f"Normalizing features...\n")
    normalized_features = F.normalize(feature_tensor, p=2, dim=1)
    
    max_batch_size = memoryManagement.getBatchSize(numEmbeddings, feature_tensor.shape, available_memory_gb=total_memory_gb, safe_fraction=0.8, FLOAT_BYTES=FLOAT_BYTES)
    
    # Ensure batch_size is at least 1 and not larger than the total size
    batch_size = max(1, min(max_batch_size, numEmbeddings))
        
    num_batches = math.ceil(numEmbeddings / batch_size)
    
    dissimilarity_parts_cpu = []
    
    print(f"Calculating cosine dissimilarity in batches of size {batch_size} MB\n")
    # 2. Calculate the full pairwise dissimilarity matrix
    for i in range(num_batches):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, numEmbeddings)
        
        batch_features = normalized_features[start_i:end_i]
        
        batch_dissimilarity = 1.0 - torch.matmul(batch_features, normalized_features.T)
        
        # Iterate through each row in the calculated batch
        for k in range(end_i - start_i):
            global_row_index = start_i + k
            
            # For row 'global_row_index', we only need similarities against 
            # columns 'j' where j > global_row_index (the upper triangle).
            # The slice starts at global_row_index + 1
            row_slice = batch_dissimilarity[k, global_row_index + 1:]
            
            # Move the slice to the CPU and append
            dissimilarity_parts_cpu.append(row_slice.cpu())
            
        # Cleanup GPU memory for the next batch
        del batch_features
        del batch_dissimilarity
        torch.cuda.empty_cache()

    # 3. Extract the unique pairwise similarities
    dissimilarity_array_tensor = torch.cat(dissimilarity_parts_cpu)
    np_dissimilarity = dissimilarity_array_tensor.cpu().numpy()
    

    print(f"Saving similarity array to {dissimilarity_path} to free up system memory.")
    np.save(dissimilarity_path, np_dissimilarity)
    
    if len(csvUtils.findInCsv(csv_path, ['model', 'model_source', 'model_weights', 'dataset'], [modelc.name, modelc.source, modelc.weights, dt_name_w_subset])) == 0:
        csvUtils.writeCsvLine(csv_path, [modelc.name, modelc.source, modelc.weights, dt_name_w_subset, dissimilarity_path])
    
    # Delete the large tensor from RAM immediately after saving
    del dissimilarity_array_tensor
    del np_dissimilarity
    return dissimilarity_path

def calculateCorrelations(path_a, path_b, correlation_type='spearman', chunked=False):
    print("\nLoading similarity arrays from disk for correlation analysis...\n")
    array_a = None
    array_b = None
    
    try:
        print(f"\n--- Starting {correlation_type} Correlation Calculation ---")
        
        print("\nLoading similarity arrays from disk...\n")
        
        a = np.load(path_a)
        b = np.load(path_b)
        
        if a.shape != b.shape:
            raise ValueError("Arrays must match in size.")
        
        if correlation_type == 'pearson':
            print("Calculating Pearson...")
            r, _ = pearsonr(a, b)
            return r, None
        
        elif correlation_type == 'spearman':
            print("Calculating Spearman...")
            if chunked:
                
                chunk_size = memoryManagement.getBatchSize(len(a), a.shape, available_memory_gb=32)
                
                r, _ = similarityUtils.chunkedSpearman(path_a, path_b, chunk_size=chunk_size)
            
            else:
                r, _ = spearmanr(a, b)
                
            return r, None

        else:
            raise ValueError("correlation_type must be 'pearson' or 'spearman'.")

    except Exception as e:
        print(f"ERROR: {e}")
        raise
    
def getCosineDissimilarity(ptPath, dissimilarity_csv, dissimilarity_folder, modelc: Model, dt_info : DtInfo, existing_dissimilarity=False):
    
    m_name = modelc.name.replace('/','-')
    m_weights = modelc.weights.replace('/','-')
    s_name = modelc.source.replace('/','-')
    
    ans = isDissimilarityCalculated(dt_info.name_w_subset, dissimilarity_csv, modelc)
    
    if existing_dissimilarity and (len(ans) > 0):
        return ans[0]['path']
    else:
        dissimilarity_path = os.path.join(dissimilarity_folder, f"{m_name}_{m_weights}_{s_name}_{dt_info.name_w_subset}.npy")
        cosineDissimilarity(ptPath, dissimilarity_csv, dissimilarity_path, modelc, dt_info.name_w_subset)
        return dissimilarity_path

def isDissimilarityCalculated(dt_name_w_subset, dissimilarity_csv, modelc):
    
    params = ['model', 'model_source', 'model_weights', 'dataset']
    values = [modelc.name, modelc.source, modelc.weights, dt_name_w_subset]
    
    ans = csvUtils.findInCsv(dissimilarity_csv, params, values)
    
    return ans