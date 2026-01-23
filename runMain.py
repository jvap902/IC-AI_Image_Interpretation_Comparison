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
        ('clip', 'ViT-B/32', 'DEFAULT'),
        ('clip', 'ViT-B/16', 'DEFAULT'),
        ('clip', 'ViT-L/14', 'DEFAULT'),
        ('open_clip', 'ViT-B-32-256', 'DEFAULT'),
        ('open_clip', 'ViT-B-16', 'DEFAULT'),
        ('open_clip', 'ViT-L-14', 'DEFAULT'),
        ('huggingface', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'DEFAULT'),
        ('huggingface', 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'DEFAULT'),
        ('torchvision', 'resnet18', 'IMAGENET1K_V1'),
        ('torchvision', 'resnet50', 'IMAGENET1K_V1'),
        ('torchvision', 'resnet152', 'IMAGENET1K_V1'),
        ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V1'),
        ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_V2'),
        ('torchvision', 'regnet_y_16gf', 'IMAGENET1K_SWAG_E2E_V1'),
        ('torchvision', 'regnet_y_32gf', 'IMAGENET1K_V2'),
        ('torchvision', 'vit_b_16', 'IMAGENET1K_V1'),
        ('torchvision', 'vit_b_16', 'IMAGENET1K_SWAG_E2E_V1'),
        ('torchvision', 'vit_l_16', 'IMAGENET1K_V1'),
        ('torchvision', 'vit_h_14', 'IMAGENET1K_SWAG_E2E_V1'),
        ('torchvision', 'maxvit_t', 'IMAGENET1K_V1'),
        ('torchvision', 'convnext_tiny', 'IMAGENET1K_V1'),
        ('torchvision', 'convnext_base', 'IMAGENET1K_V1'),
        ('torchvision', 'swin_t', 'IMAGENET1K_V1'),
        ('torchvision', 'swin_v2_t', 'IMAGENET1K_V1'),
        ('torchvision', 'efficientnet_b0', 'IMAGENET1K_V1'),
        ('torchvision', 'efficientnet_b4', 'IMAGENET1K_V1'),
        ('torchvision', 'efficientnet_b7', 'IMAGENET1K_V1')
        
    ]
    
    '''
    (src1, model1, weight1) = instances[13]
    (src2, model2, weight2) = instances[15]
    
    print(f"    --- Running test: {model1} ({weight1}) x {model2} ({weight2}) ---")
    
    arguments_to_pass = ["--dataset", "timm/mini-imagenet", "--m1_source", src1, "-m1", model1, "--m1_weights", weight1, 
                             "--m2_source", src2, "-m2", model2, "--m2_weights", weight2]
    run_main_with_subprocess(arguments_to_pass)
    '''
    
    datasets = ['timm/mini-imagenet', 'imagenet-sketch', 'cifar10', 'cifar100']
    
    #begin = 0
    #(src1, model1, weight1) = instances[begin]
    #(src2, model2, weight2) = instances[4]
    #for dataset in datasets:
    #    dt_name = dataset.replace('/', '-')
    #    
    #    print(f"    --- Running {dataset} test: {model1} ({weight1}) x {model2} ({weight2}) ---")
    #
    #    arguments_to_pass = ["--dataset", dataset, "--m1_source", src1, "-m1", model1, "--m1_weights", weight1, 
    #                        "--m2_source", src2, "-m2", model2, "--m2_weights", weight2, "-ed", "-out", f"./ztempData/{dt_name}Data.csv"]
    #    run_main_with_subprocess(arguments_to_pass)
    
    begin = 0
    (src1, model1, weight1) = instances[begin]
    for (src2, model2, weight2) in instances[18:]:
        for dataset in datasets:
            dt_name = dataset.replace('/', '-')
            
            print(f"    --- Running {dataset} test: {model1} ({weight1}) x {model2} ({weight2}) ---")
        
            arguments_to_pass = ["--dataset", dataset, "--m1_source", src1, "-m1", model1, "--m1_weights", weight1, 
                                "--m2_source", src2, "-m2", model2, "--m2_weights", weight2, "-ed", "-out", f"./ztempData/{dt_name}Data.csv"]
            run_main_with_subprocess(arguments_to_pass)


    begin = 0
    for idx, (src1, model1, weight1) in enumerate(instances[begin:7]):
        for (src2, model2, weight2) in instances[idx+begin+1:]:
            for dataset in datasets:
                dt_name = dataset.replace('/', '-')
        
                print(f"    --- Running {dataset} test: {model1} ({weight1}) x {model2} ({weight2}) ---")
            
                arguments_to_pass = ["--dataset", dataset, "--m1_source", src1, "-m1", model1, "--m1_weights", weight1, 
                                    "--m2_source", src2, "-m2", model2, "--m2_weights", weight2, "-ed", "-out", f"./ztempData/{dt_name}Data.csv"]
                run_main_with_subprocess(arguments_to_pass)