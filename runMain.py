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
    command = [sys.executable, '-m', "src.core.main"] + args
    
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

class startParams(TypedDict):
    fst_instance: int
    snd_instance: int
    dataset: int
    interrupt: Tuple[int, int, int] #índices do: dataset, modelo1, modelo2
    

def rsa_args(run_data):
    dataset, subset, src1, model1, weight1, src2, model2, weight2, dt_name = run_data
    
    print(f"    --- Running {dataset} RSA: {model1} ({weight1}) x {model2} ({weight2}) ---")
    
    sleep(5.0)
    
    return ["--dataset", dataset, "--specific_subset", str(subset), "-ndsc", "--m1_source", src1, "-m1", model1, "--m1_weights", weight1, '-met', 'rsa',
                                    "--m2_source", src2, "-m2", model2, "--m2_weights", weight2, "-ed", "--no_validation", "-out", f"./dataStorage/rsaData/{dt_name}Data.csv"]
    
def cka_args(run_data):
    dataset, subset, src1, model1, weight1, src2, model2, weight2, dt_name = run_data
    
    sleep(5.0)
    
    print(f"    --- Running {dataset} CKA: {model1} ({weight1}) x {model2} ({weight2}) ---")
    
    return ["--dataset", dataset, "--specific_subset", str(subset), "-ndsc", "--m1_source", src1, "-m1", model1, "--m1_weights", weight1, 
                                    "--m2_source", src2, "-m2", model2, "--m2_weights", weight2, '-met', 'cka']
        
def run(instances, datasets, method, start_params: startParams, interrupt=True):
    
    begin = start_params['fst_instance']
    snd_start = start_params['snd_instance']
    initial_dt = start_params['dataset']
    
    (src1, model1, weight1) = instances[begin]
    (src2, model2, weight2) = instances[snd_start]
    
    m1_interr = True if (src1, model1, weight1) == instances[start_params["interrupt"][1]] else False
    m2_interr = True if (src2, model2, weight2) == instances[start_params["interrupt"][2]] else False
    
    for (dataset, subset) in datasets[initial_dt:]:
        dt_interr = True if (dataset, subset) == datasets[start_params["interrupt"][0]] else False
        
        if interrupt and m1_interr and m2_interr and dt_interr:
            return
        
        dt_name = dataset.replace('/', '-')
        run_data = [dataset, subset, src1, model1, weight1, src2, model2, weight2, dt_name]
        
        arguments = method(run_data)
        
        run_main_with_subprocess(arguments)
        
    
    snd_start = snd_start + 1
    
    for (src2, model2, weight2) in instances[snd_start:]:
        m2_interr = True if (src2, model2, weight2) == instances[start_params["interrupt"][2]] else False
        for (dataset, subset) in datasets:
            dt_interr = True if (dataset, subset) == datasets[start_params["interrupt"][0]] else False
            
            if interrupt and m1_interr and m2_interr and dt_interr:
                return
            
            dt_name = dataset.replace('/', '-')
            
            run_data = [dataset, subset, src1, model1, weight1, src2, model2, weight2, dt_name]
                    
            arguments = method(run_data)
            
            run_main_with_subprocess(arguments)


    begin = begin+1
    for idx, (src1, model1, weight1) in enumerate(instances[begin:]):
        m1_interr = True if (src1, model1, weight1) == instances[start_params["interrupt"][1]] else False
        for (src2, model2, weight2) in instances[idx+begin+1:]:
            m2_interr = True if (src2, model2, weight2) == instances[start_params["interrupt"][2]] else False
            for (dataset, subset) in datasets:
                dt_interr = True if (dataset, subset) == datasets[start_params["interrupt"][0]] else False
            
                if interrupt and m1_interr and m2_interr and dt_interr:
                    return
                
                dt_name = dataset.replace('/', '-')
        
                run_data = [dataset, subset, src1, model1, weight1, src2, model2, weight2, dt_name]
            
                arguments = method(run_data)
                
                run_main_with_subprocess(arguments)


def revalidate(instances, datasets, start_params: startParams):
    
    for i in range(len(instances)):
        for (dataset, subset) in datasets:
            dt_name = dataset.replace('/', '-')
        
            idx1, (s1, m1, w1) = instances.index(instances[i]), instances[i]
            idx2, (s2, m2, w2) = instances.index(instances[len(instances)-1-i]), instances[len(instances)-1-i]
            
            if idx1 > idx2:
                return
            
            args = ["--dataset", dataset, "--specific_subset", str(subset), "-ndsc", "--m1_source", s1, "-m1", m1, "--m1_weights", w1, 
                                    "--m2_source", s2, "-m2", m2, "--m2_weights", w2, '-met', 'validation']
        
            run_main_with_subprocess(args)

if __name__ == "__main__":

    instances = config.instances
    
    datasets = config.datasets #do artigo: sketch, cifar10, aircraft, imagenet-1k
    datasets = [datasets[9]] #jpeg_compression 5
    
    datasets = [('imagenet-c-motion_blur-5', 0)]
    
    method_name = 'rsa'
    
    match method_name:
        case 'rsa':
            method = rsa_args
        case 'cka':
            method = cka_args
        case _:
            raise
        
    fst_idx = 0#codToInstance(14, 'a')[0]
    snd_idx = codToInstance(12, 'a')[0]+1
    
    fst_model_interr = 1#codToInstance(13, 'b')[0]
    snd_model_interr = codToInstance(17, 'a')[0]
    
    start_params = startParams({'fst_instance': fst_idx, 'snd_instance': snd_idx, 'dataset': 0, 'interrupt': (0, fst_model_interr, snd_model_interr)})
    
    run(instances, datasets, method, start_params, interrupt=False)
    #revalidate(instances, datasets, method, start_params)


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