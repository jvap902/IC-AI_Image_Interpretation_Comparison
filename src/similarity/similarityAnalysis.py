import torch.nn.functional as F
import torch
import numpy as np
import math
from scipy.stats import pearsonr
from . import similarityUtils

device = "cuda" if torch.cuda.is_available() else "cpu"

total_memory_gb = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

def cosineSimilarity(ptPath, save_path): #retorna vetor de similaridades de cosseno
    
    print(f"Loding feature tensor for cosine similarity \n")
    
    feature_tensor = torch.load(ptPath, map_location='cpu', weights_only=True).to(device)
    numEmbeddings = feature_tensor.size(0)

    print(f"Feature tensor loaded with shape: {feature_tensor.shape}\n")
    print(f"Normalizing features...\n")
    normalized_features = F.normalize(feature_tensor, p=2, dim=1)


    # Constants for memory calculation (assuming torch.float32)
    FLOAT_BYTES = 4 
    
    # Use a safe fraction of the available VRAM in bytes (80% of the allocated GB)
    available_bytes = total_memory_gb * 1024**3 * 0.8  
    
    max_batch_size = math.floor(available_bytes / (numEmbeddings * FLOAT_BYTES))
    
    # Ensure batch_size is at least 1 and not larger than the total size
    batch_size = max(1, min(max_batch_size, numEmbeddings))
        
    num_batches = math.ceil(numEmbeddings / batch_size)
    
    similarity_parts_cpu = []
    
    print(f"Calculating cosine similarity in batches of size {batch_size} MB\n")
    # 2. Calculate the full pairwise similarity matrix
    for i in range(num_batches):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, numEmbeddings)
        
        batch_features = normalized_features[start_i:end_i]
        
        batch_similarity = torch.matmul(batch_features, normalized_features.T)
        
        # Iterate through each row in the calculated batch
        for k in range(end_i - start_i):
            global_row_index = start_i + k
            
            # For row 'global_row_index', we only need similarities against 
            # columns 'j' where j > global_row_index (the upper triangle).
            # The slice starts at global_row_index + 1
            row_slice = batch_similarity[k, global_row_index + 1:]
            
            # Move the slice to the CPU and append
            similarity_parts_cpu.append(row_slice.cpu())
            
        # Cleanup GPU memory for the next batch
        del batch_features
        del batch_similarity
        torch.cuda.empty_cache()

    # 3. Extract the unique pairwise similarities
    similarity_array_tensor = torch.cat(similarity_parts_cpu)#similarity_matrix[torch.triu(torch.ones(numEmbeddings, numEmbeddings, dtype=torch.bool), diagonal=1)]
    
    if save_path:
            print(f"Saving similarity array to {save_path} to free up system memory.")
            torch.save(similarity_array_tensor, save_path)
            # Delete the large tensor from RAM immediately after saving
            del similarity_array_tensor
            return save_path
    
    return np.array(similarity_array_tensor)

def calculateCorrelations(path_a, path_b, correlation_type='spearman'):
    print("\nLoading similarity arrays from disk for correlation analysis...\n")
    array_a = None
    array_b = None
    
    try:
        print(f"\n--- Starting {correlation_type} Correlation Calculation ---")
        
        print("\nLoading similarity arrays from disk...\n")
        
        a = similarityUtils.loadTensorAsNP(path_a)
        b = similarityUtils.loadTensorAsNP(path_b)
        
        if a.shape != b.shape:
            raise ValueError("Arrays must match in size.")
        
        if correlation_type == 'pearson':
            print("Calculating Pearson...")
            r, _ = pearsonr(a, b)
            return r, None
        
        elif correlation_type == 'spearman':
            print("Calculating Spearman (chunked)...")
            r, _ = similarityUtils.chunkedSpearman(path_a, path_b)
            return r, None

        else:
            raise ValueError("correlation_type must be 'pearson' or 'spearman'.")

    except Exception as e:
        print(f"ERROR: {e}")
        raise