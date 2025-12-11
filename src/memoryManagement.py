import torch

def aggressive_cleanup(tensors_to_delete):
    """Aggressively deletes tensors and clears CUDA cache."""
    print("Performing aggressive memory cleanup...")
    for tensor in tensors_to_delete:
        if tensor is not None:
            # Need to explicitly check if it's a torch tensor to call .storage()
            if isinstance(tensor, torch.Tensor):
                # Ensure the storage is deallocated, especially for large CPU tensors
                if tensor.is_shared():
                    tensor.storage().unref()
            del tensor
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Forces Python's garbage collector to run
    import gc
    gc.collect()
    print("Cleanup complete.")