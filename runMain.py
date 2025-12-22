import subprocess
import sys
from src import *

def run_main_with_subprocess(args):
    """
    Executes the main.py script as an external command, passing along 
    any command-line arguments provided to this script.
    """
    
    # Base command: [python interpreter, 'main.py']
    command = [sys.executable, 'main.py'] + args
    
    try:
        # subprocess.run waits for the command to complete
        result = subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("Output from called_script.py:")
        print(result.stdout)

        if result.returncode != 0:
            print(f"Process exited with code {result.returncode}")
            
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")


if __name__ == "__main__":
    # Capture all command-line arguments passed to this script, 
    # excluding the script name itself (sys.argv[0] is 'run_pipeline.py').
    
    instances = [
        ('resnet18', 'IMAGENET1K_V1'),
        ('resnet50', 'IMAGENET1K_V1'),
        ('resnet152', 'IMAGENET1K_V1'),
        ('convnext_tiny', 'IMAGENET1K_V1'),
        ('convnext_base', 'IMAGENET1K_V1'),
        ('regnet_y_16gf', 'IMAGENET1K_V1'),
        ('regnet_y_16gf', 'IMAGENET1K_V2'),
        ('regnet_y_16gf', 'IMAGENET1K_SWAG_E2E_V1'),
        ('regnet_y_32gf', 'IMAGENET1K_V2'),
        ('vit_b_16', 'IMAGENET1K_V1'),
        ('vit_b_16', 'IMAGENET1K_SWAG_E2E_V1'),
        ('vit_l_16', 'IMAGENET1K_V1'),
        ('vit_h_14', 'IMAGENET1K_SWAG_E2E_V1'),
        ('swin_t', 'IMAGENET1K_V1'),
        ('swin_v2_t', 'IMAGENET1K_V1'),
        ('maxvit_t', 'IMAGENET1K_V1'),
        ('efficientnet_b0', 'IMAGENET1K_V1'),
        ('efficientnet_b4', 'IMAGENET1K_V1'),
        ('efficientnet_b7', 'IMAGENET1K_V1')
        
    ]

    for idx, (model1, weight1) in enumerate(instances[1:]):
        for (model2, weight2) in instances[idx+1:]:
        
            print(f"    --- Running test: {model1} x {model2} ---")
        
            arguments_to_pass = ["--dataset", "timm/mini-imagenet", "--m1_source", "torchvision", "-m1", model1, "--m1_weights", weight1, 
                                 "--m2_source", "torchvision", "-m2", model2, "--m2_weights", weight2]
            run_main_with_subprocess(arguments_to_pass)