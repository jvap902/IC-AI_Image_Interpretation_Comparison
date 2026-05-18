import seaborn
import matplotlib as plt

def heatmap(df, save_path=None, show=True):
    
    plt.figure(figsize=(10, 8))
    
    ax = seaborn.heatmap(df, vmin=-0.5, vmax=1.0)

    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(pad=0.8)
    if save_path != None: plt.savefig(save_path, dpi=100)
    if show: plt.show()