import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    pass