import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import csv


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
        embedding_tensor = torch.load(pt_path)
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



if __name__ == '__main__':
    pass