import sys
import subprocess
from time import sleep
from pathlib import Path
from typing import TypedDict, Tuple
from src import *
from src.codifications import *
from utils.debug import *

def run_main_with_subprocess(args):
    """
    Executes the main.py script as an external command, passing along 
    any command-line arguments provided to this script.
    """
    
    # Base command: [python interpreter, 'main.py']
    command = [sys.executable, '-m', "src.dataAnalysis.main"] + args
    
    try:
        # subprocess.run waits for the command to complete
        result = subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("Output from called_script.py:")
        print(result.stdout)

        if result.returncode != 0:
            print(f"Process exited with code {result.returncode}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n[!] CRITICAL ERROR: Subprocess failed with return code {e.returncode}")
        print(f"Arguments: {' '.join(args)}")
        # This line stops the entire script immediately
        sys.exit(1)


if __name__ == "__main__":

    instances = config.instances
    
    datasets_names = ['imagenet-c']
    
    datasets = [idx for idx, dt_sb in enumerate(config.datasets)]# if any(name in dt_sb[0] for name in datasets_names)]
    met = 'cka'
    analysis = 'basic'
    save_dir = "dataStorage/processedResults/cka"
    
    for dt_idx in datasets:
        save_path = f'{save_dir}/{config.datasets[dt_idx][0].replace('/', '-')}.png'
        
        args = ['-a', analysis, '-met', met, '-d', str(dt_idx), '-g', '-s', save_path]
    
        run_main_with_subprocess(args)


#1e - ('huggingface', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'DEFAULT')
#2e - ('huggingface', 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'DEFAULT')
#3d - ('clip', 'ViT-B/32', 'DEFAULT')
#4d - ('clip', 'ViT-B/16', 'DEFAULT')
#5d - ('clip', 'ViT-L/14', 'DEFAULT')
#6d - ('open_clip', 'ViT-B-32-256', 'DEFAULT')
#7d - ('open_clip', 'ViT-B-16', 'DEFAULT')
#8d - ('open_clip', 'ViT-L-14', 'DEFAULT')
#9a - ('torchvision', 'resnet18', 'IMAGENET1K_V1')
#10a - ('torchvision', 'resnet50', 'IMAGENET1K_V1')
#11a - ('torchvision', 'resnet152', 'IMAGENET1K_V1')
#12a - ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V1')
#12b - ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V2')
#12c - ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_SWAG_E2E_V1')
#13b - ('torchvision', 'regnet_y_32gf', 'IMAGENET1K_V2')
#14a - ('torchvision', 'vit_b_16', 'IMAGENET1K_V1')
#14c - ('torchvision', 'vit_b_16', 'IMAGENET1K_SWAG_E2E_V1')
#15a - ('torchvision', 'vit_l_16', 'IMAGENET1K_V1')
#16c - ('torchvision', 'vit_h_14', 'IMAGENET1K_SWAG_E2E_V1')
#17a - ('torchvision', 'maxvit_t', 'IMAGENET1K_V1')
#18a - ('torchvision', 'convnext_tiny', 'IMAGENET1K_V1')
#19a - ('torchvision', 'convnext_base', 'IMAGENET1K_V1')
#20a - ('torchvision', 'swin_t', 'IMAGENET1K_V1')
#21a - ('torchvision', 'swin_v2_t', 'IMAGENET1K_V1')
#22a - ('torchvision', 'efficientnet_b0', 'IMAGENET1K_V1')
#23a - ('torchvision', 'efficientnet_b4', 'IMAGENET1K_V1')
#24a - ('torchvision', 'efficientnet_b7', 'IMAGENET1K_V1')