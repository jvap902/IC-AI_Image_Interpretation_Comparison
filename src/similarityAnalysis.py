import torch.nn.functional as F
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
    # 2. Calculate the full pairwise similarity matrix
    # The dot product of two normalized vectors is their cosine similarity.
    # [N x D] @ [D x N] = [N x N] similarity matrix
    # torch.transpose(normalized_features, 0, 1) is equivalent to normalized_features.T
    similarity_matrix = torch.matmul(normalized_features, normalized_features.T)

    # 3. Extract the unique pairwise similarities
    # We take the upper triangular part (k=1 excludes the main diagonal, which is all 1.0)
    # The output is a 1D tensor containing the unique pairs.
    similarity_array_tensor = similarity_matrix[torch.triu(torch.ones(numEmbeddings, numEmbeddings, dtype=torch.bool), diagonal=1)]

    # 4. Convert the final result to a NumPy array for final analysis
    # This single transfer from GPU (if used) to CPU is extremely efficient.
    similarity_array = similarity_array_tensor.cpu().numpy()
    
    # array_size = numEmbeddings * (numEmbeddings-1) // 2

    # similarity_array = [0 for _ in range(array_size)]

    # print(f"Calculating values\n")
    # pos = 0
    # for i in range(numEmbeddings):
    #     for j in range(i+1, numEmbeddings):
    #         sim = F.cosine_similarity(feature_tensor[i].unsqueeze(0), feature_tensor[j].unsqueeze(0), dim=1)

    #         similarity_array[pos] = sim.item()
            
    #         pos += 1
            

    return np.array(similarity_array)