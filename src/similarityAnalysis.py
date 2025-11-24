import torch.nn.functional as F
import torch
import numpy as np

def similarityVectorIndex(first, second, numEmbeddings):
    if second > first:
        first, second = second, first

    temp = 0

    fst = 0

    while(temp < first):
        fst += numEmbeddings - fst

    return fst + (second - first)
    

    

def cosineSimilarity(ptPath): #retorna vetor de similaridades de cosseno
    feature_tensor = torch.load(ptPath)
    
    numEmbeddings = feature_tensor.size(0)

    array_size = numEmbeddings**2 - numEmbeddings

    similarity_array = [0 for _ in range(array_size)]

    pos = 0
    for i in range(numEmbeddings):
        for j in range(i+1, numEmbeddings):
            sim = F.cosine_similarity(feature_tensor[i].unsqueeze(0), feature_tensor[j].unsqueeze(0), dim=1)

            similarity_array[pos] = sim.item()
            
            pos += 1
            

    return np.array(similarity_array)