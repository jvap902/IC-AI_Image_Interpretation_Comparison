import torch.nn.functional as F
import torch
import numpy as np
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

def similarityVectorIndex(first, second, numEmbeddings):
    if second > first:
        first, second = second, first

    temp = 0

    fst = 0

    while(temp < first):
        fst += numEmbeddings - fst

    return fst + (second - first)    

def cosineSimilarity(ptPath): #retorna vetor de similaridades de cosseno
    
    print(f"Loding feature tensor for cosine similarity \n")
    
    feature_tensor = torch.load(ptPath).to(device)
    numEmbeddings = feature_tensor.size(0)

    normalized_features = F.normalize(feature_tensor, p=2, dim=1)

    batch_size = total_memory // (numEmbeddings * feature_tensor.size(1) * 4 * 4) # Rough estimate
    num_batches = math.ceil(numEmbeddings / batch_size)

    # 2. Calculate the full pairwise similarity matrix
    for i in range(num_batches):
        start_i = i * batch_size
        end_i = min((i + 1) * batch_size, numEmbeddings)
        batch_features = normalized_features[start_i:end_i]
        batch_similarity = torch.matmul(batch_features, normalized_features.T)
        if i == 0:
            similarity_matrix = batch_similarity
        else:
            similarity_matrix = torch.cat((similarity_matrix, batch_similarity), dim=0)

    # 3. Extract the unique pairwise similarities
    similarity_array_tensor = similarity_matrix[torch.triu(torch.ones(numEmbeddings, numEmbeddings, dtype=torch.bool), diagonal=1)]

    # 4. Convert the final result to a NumPy array for final analysis
    similarity_array = similarity_array_tensor.cpu().numpy()
    
    return np.array(similarity_array)