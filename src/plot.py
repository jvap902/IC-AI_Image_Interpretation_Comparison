import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import seaborn
import pandas as pd
import ast

def plot_history(history):
    """Plots the training and validation history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation accuracy values
    ax1.plot(history['train_acc'])
    ax1.plot(history['val_acc'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'val'], loc='upper left')

    # Plot training & validation loss values
    ax2.plot(history['train_loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['train', 'val'], loc='upper left')

    plt.show()

def plot_pt_file(file_path):
    """
    Loads a PyTorch tensor and writes its string representation line-by-line 
    to a .txt file, which is more robust for very large arrays.

    Args:
        file_path (str): The path to the input .pt file.
    """
    try:
        pt_path = Path(file_path)
        
        if not pt_path.exists():
            print(f"Error: Input file not found at {file_path}")
            return

        txt_path = pt_path.with_suffix(".txt")
        
        # 1. Load the tensor
        embedding_tensor = torch.load(pt_path, weights_only=False)
        np_array = embedding_tensor.cpu().numpy()

        print(f"Successfully loaded tensor with shape: {embedding_tensor.shape}")

        # 2. Open the file and write row by row
        print(f"Starting to write {np_array.shape[0]} rows to {txt_path}. This may take a moment...")
        
        # Use np.savetxt for simplicity and robustness in writing large arrays, 
        # which handles the internal looping and formatting efficiently.
        # We explicitly set newline='\n' for cross-platform compatibility.
        np.savetxt(
            txt_path, 
            np_array, 
            fmt='%.6f',  # Format to 6 decimal places (you can adjust this)
            delimiter=', ', 
            newline='\n'
        )

        print(f"Visualization array saved successfully to: {txt_path}")

    except Exception as e:
        print(f"An unexpected error occurred in plot_pt_file: {e}")

def similarityCsv(similarity_array, file_path, num_embeddings, model_name):

    print(f"\nconstruíndo {file_path}\n")

    current_first_id = 0
    current_second_id = 1
    
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(['model', 'first_embedding_id', 'second_embedding_id', 'cosine_correlation']) #colunas

        for cosine in similarity_array:
            if(current_second_id > num_embeddings-1):
                current_first_id += 1
                current_second_id = current_first_id + 1

            csvwriter.writerow([model_name, str(current_first_id), str(current_second_id), str(cosine)])

            current_second_id += 1

def writeCsvLine(file_path, data):
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(data)

def read_csv_to_array(filepath):
    data = []
    with open(filepath, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data

def correlationGraphic(filepath, xlabel, ylabel, subplot_fields=[]):
    
    try: 
        data = read_csv_to_array(filepath)
        
        i=0
        
        num_subplots = len(subplot_fields)
        
        while True: # vai ser usado como um do-while
            i += 1
            
            if (num_subplots):
                yfield = subplot_fields[i-1]
            else:
                yfield = ylabel
        
            # acha quais colunas se quer usar
            for name in data[0]:
                if (name == xlabel):
                    xpos = data[0].index(name)
                if (name == yfield):
                    ypos = data[0].index(name)
            
            xelements = []
            yelements = []
            
            # coloca em vetores numéricos os valores para plot
            for row in data[1:]: #tirando linha dos títulos
                xelements.append(np.float64(row[xpos]))
                yelements.append(np.float64(row[ypos]))
                
            plt.plot(xelements, yelements, label = yfield)
            
              
            if i >= num_subplots:
                break
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.legend()
        plt.show()
    
    
    except Exception as e:
        print(f"An unexpected error occurred in correlationGraphic: {e}")

def findInCsv(file_path, params, values):
    if(len(params) != len(values)):
        raise ValueError("The number of parameters should be the as the number of values serched")
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = list(csv.DictReader(file))
        
        ans = []
        
        for row in reader:
            all_equal = True
            
            for idx, param in enumerate(params):
                if (str(values[idx]) != row[param]):
                    all_equal = False
                    break
            
            if all_equal:
                ans.append(row)
                
    return ans

def getStringIntArray(string):
    
    string = string[1:-1]
    ans = [int(num) for num in string.split(', ')]
    
    return ans

def getStringStrArray(string):    
    
    return ast.literal_eval(string)

def paramDataFrameFromCsv(csv_path, param, diagonal=1.0):
    
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = list(csv.DictReader(file))
        
        return dataFrameFromData(reader, param, diagonal=diagonal)
        
        #ainda não testado totalmente se pode ser substituído, portanto código antigo está abaixo
        if not reader:
            return np.array([])

        # 1. Identificar todos os modelos únicos para saber o tamanho da matriz
        models = []
        for row in reader:
            if (row['fst_model_source'], row['first_model'], row['fst_weights']) not in models:
                models.append((row['fst_model_source'], row['first_model'], row['fst_weights']))
            if (row['snd_model_source'], row['second_model'], row['snd_weights']) not in models:
                models.append((row['snd_model_source'], row['second_model'], row['snd_weights']))
        
        n = len(models)
        
        print(n)
        
        # Criar matriz preenchida com zeros (ou 1.0 na diagonal)
        matrix = np.zeros((n, n))
        np.fill_diagonal(matrix, 1.0)

        # Criar um mapeamento de nome do modelo para índice (0, 1, 2...)
        model_to_idx = {model: i for i, model in enumerate(models)}

        # 2. Preencher a matriz com os valores do CSV
        for row in reader:
            m1 = (row['fst_model_source'], row['first_model'], row['fst_weights'])
            m2 = (row['snd_model_source'], row['second_model'], row['snd_weights'])
            val = np.float32(row[param])
            
            i, j = model_to_idx[m1], model_to_idx[m2]
            matrix[i][j] = val
            matrix[j][i] = val # Garante a simetria se o CSV tiver apenas um lado
    
        dataFrame = pd.DataFrame(matrix, columns=models, index=models)

    return dataFrame

def dataFrameFromData(data, param, diagonal=1.0):
    if not data:
        return np.array([])

    # 1. Identificar todos os modelos únicos para saber o tamanho da matriz
    raw_items = []
    for row in data:
        raw_items.append((row['fst_model_source'], row['first_model'], row['fst_weights']))
        raw_items.append((row['snd_model_source'], row['second_model'], row['snd_weights']))
    
    # dict.fromkeys para remover duplicatas preservando a ordem
    models = list(dict.fromkeys(raw_items))
    
    n = len(models)
    
    # Criar matriz preenchida com zeros (ou 1.0 na diagonal)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, diagonal)

    # Criar um mapeamento de nome do modelo para índice (0, 1, 2...)
    model_to_idx = {model: i for i, model in enumerate(models)}

    # 2. Preencher a matriz com os valores do CSV
    for row in data:
        m1 = (row['fst_model_source'], row['first_model'], row['fst_weights'])
        m2 = (row['snd_model_source'], row['second_model'], row['snd_weights'])
        val = np.float32(row[param])
        
        i, j = model_to_idx[m1], model_to_idx[m2]
        matrix[i][j] = val
        matrix[j][i] = val # Garante a simetria se o CSV tiver apenas um lado

    dataFrame = pd.DataFrame(matrix, columns=models, index=models)

    return dataFrame
            
def heatMap(csv_path, correlation_type):
    data = paramDataFrameFromCsv(csv_path, correlation_type)
    
    print(data.shape)
    plt.figure(figsize=(data.shape[1] * 0.8, data.shape[0] * 0.4))
    
    seaborn.heatmap(data, vmin=-0.5, vmax=1.0)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pass