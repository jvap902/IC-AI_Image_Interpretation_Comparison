import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as scipy

def heatmap(df, save_path=None, show=True):
    
    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(df, vmin=-0.5, vmax=1.0)

    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(pad=0.8)
    if save_path != None: plt.savefig(save_path, dpi=100)
    if show: plt.show()
    
def dendrogram(Z, labels, title=None, ylabel=None, xlabel=None, dpi=100, show=True, save_path = None):
    
    plt.figure(figsize=(8, 10))
    plt.rcParams['lines.linewidth'] = 2.5
    scipy.dendrogram(
        Z,
        orientation='right',
        labels=labels,
        leaf_rotation=0,
        leaf_font_size=14,
        color_threshold=None # Can be set to a float to color clusters at a specific depth
    )
    
    label_font_size = 18
    plt.xlim(0.0, 0.5)
    plt.xticks(np.arange(0.0, 0.5, 0.05))
    plt.tick_params('x', labelsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.2)
    if title: plt.title(title, fontsize=label_font_size)
    if ylabel: plt.ylabel(ylabel, fontsize=label_font_size)
    if xlabel: plt.xlabel(xlabel, fontsize=label_font_size)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=dpi)
    if show: plt.show()
    
def barChart(df, bar_param, save_path=None, dpi=100, ylabel=None, xlabel=None, show=True):
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(df, x='Model', y=f'{bar_param}_avg', hue=f'{bar_param}_avg', legend=False, palette=sns.color_palette("flare_r", as_cmap=True), saturation=1.0)
    plt.yticks(np.arange(0.0, 1.0, 0.1))
    if xlabel: ax.set(xlabel=xlabel)
    if ylabel: ax.set(ylabel=ylabel)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=dpi)
    if show: plt.show()