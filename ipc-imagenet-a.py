from pathlib import Path
from src.plot import writeCsvLine

def get_immediate_subdirectories_pathlib(parent_dir):
    p = Path(parent_dir)
    # Use .name to get just the folder name string, not the full Path object
    return [x.name for x in p.iterdir() if x.is_dir()]


def count_files_in_folder(folder_path):
    """
    Counts the number of files directly within a specified folder (non-recursive).
    """
    p = Path(folder_path)
    if not p.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory or does not exist.")
        return 0

    # Count only items that are files within the directory
    count = sum(1 for entry in p.iterdir() if entry.is_file())
    return count

# --- Example Usage ---
directory_to_scan = './data/imagenet-a' # Counts files in the current working directory
classes = get_immediate_subdirectories_pathlib(directory_to_scan)

for c in classes:
    sub_folder = directory_to_scan + f'/{c}'
    file_count = count_files_in_folder(sub_folder)

    writeCsvLine('./ipc-imagenet-a.csv', [c, file_count])
