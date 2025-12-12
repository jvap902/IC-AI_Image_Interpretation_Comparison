import torch
from scipy.stats import pearsonr, rankdata, spearmanr
import numpy as np
import tempfile
import heapq
import os

def loadTensorAsNP(path):
    tensor = torch.load(path, map_location='cpu', weights_only=True)
    return tensor.detach().cpu().numpy()

def similarityVectorIndex(first, second, numEmbeddings):
    if second > first:
        first, second = second, first

    temp = 0

    fst = 0

    while(temp < first):
        fst += numEmbeddings - fst

    return fst + (second - first)

def chunkedSpearman(path_a, path_b, chunk_size=8000000):
    tmp_a, n = write_chunks(path_a, chunk_size)
    tmp_b, _ = write_chunks(path_b, chunk_size)

    ranks_a = compute_ranks(tmp_a, n)
    ranks_b = compute_ranks(tmp_b, n)

    for f in tmp_a + tmp_b:
        os.remove(f)

    rho = pearsonr(ranks_a, ranks_b)
    return rho

def write_chunks(path, chunk_size=5_000_000):
    """Read tensor from disk in chunks and write sorted chunks as temporary files."""
    t = torch.load(path, map_location='cpu', weights_only=True).detach().cpu().numpy()
    n = t.size
    tmp_files = []

    for start in range(0, n, chunk_size):
        
        print(f"\nProgress: {(start/n)*100}% of items processed for chunking.")
        
        end = min(start+chunk_size, n)
        chunk = t[start:end]
        indices = np.arange(start, end)
        
        # create structured array (value, original_index)
        arr = np.zeros(chunk.size, dtype=[('value', np.float64), ('idx', np.int64)])
        arr['value'] = chunk
        arr['idx'] = indices
        
        # sort in-chunk
        arr.sort(order='value')
        
        # save chunk to temp file
        f = tempfile.NamedTemporaryFile(delete=False)
        np.save(f, arr)
        f.close()
        tmp_files.append(f.name)

    return tmp_files, n

def merge_sorted_chunks(tmp_files):
    """Yield items (value, original_idx) in globally sorted order."""
    iters = []
    for f in tmp_files:
        arr = np.load(f, mmap_mode='r')
        iters.append(iter(arr))
    
    # merge by value
    for item in heapq.merge(*iters, key=lambda x: x[0]):
        yield item
        
def compute_ranks(tmp_files, n):
            
    ranks = np.zeros(n, dtype=np.float64)
    merged = merge_sorted_chunks(tmp_files)
    
    prev_value = None
    run_start = 0
    run_items = []

    for position, item in enumerate(merged):  # position = 0..n-1
        value, idx = item
        if prev_value is None or value != prev_value:
            # finalize previous tie run
            if run_items:
                avg_rank = 0.5 * (run_start + (run_start+len(run_items)-1) + 2)  # 1-based avg
                for _, orig_idx in run_items:
                    ranks[orig_idx] = avg_rank
            # start new run
            run_start = position
            run_items = [item]
            prev_value = value
        else:
            run_items.append(item)

    # finalize last run
    if run_items:
        avg_rank = 0.5 * (run_start + (run_start+len(run_items)-1) + 2)
        for _, orig_idx in run_items:
            ranks[orig_idx] = avg_rank

    return ranks