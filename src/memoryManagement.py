import torch
import math

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
    
def getBatchSize(num_embeddings, embedding_shape, available_memory_gb=8, safe_fraction=0.8, FLOAT_BYTES=4):
    
    if len(embedding_shape) == 1:
        row_elements = 1       # 1 element per row
    else:
        row_elements = embedding_shape[1] 
    
    available_bytes = available_memory_gb * 1024**3 * 0.8
    row_size_bytes = row_elements * FLOAT_BYTES
    return math.floor(available_bytes / (num_embeddings * row_size_bytes))