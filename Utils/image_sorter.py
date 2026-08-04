from tqdm import tqdm
from pathlib import Path

def move_files(file_names, source_dir, dest_dir):
    
    pattern = f"@({'|'.join(file_names)})"
    
    for name in file_names:
        file_path = source_dir / name
        
        if file_path.is_file():
            file_path.replace(dest_dir / name)


def sort_imagenet1k_images():

    corrupted = Path("data/noise/gaussian_noise/1")

    original = Path("data/imagenet-1k/validation")

    for corr_path in tqdm(corrupted.iterdir(), desc="Sorting imagenet-1k images"):
        if corr_path.is_dir():
            print(f"Composing {corr_path.name}\n")
            
            file_names = [f.name for f in corr_path.iterdir() if f.is_file()]
            
            print(file_names[:10])
            
            new_folder = original / corr_path.name
            new_folder.mkdir(exist_ok=True)
            
            move_files(file_names, original, new_folder)            
            
if __name__ == "__main__":
    sort_imagenet1k_images()