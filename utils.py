import gc
import torch
import psutil
from tabulate import tabulate

# Function to clear memory cache
def empty_cache():
    gc.collect()
    torch.mps.empty_cache()
    print('Cache emptied: python (GC) and MPS ')

def mps_memory_usage(to_print=True):
    tensors = round(torch.mps.current_allocated_memory() / 10**6)
    total = round(torch.mps.driver_allocated_memory() / 10**6) 
    
    
    if to_print:
        print(
            'Tensors: {} (MB),\nTotal MPS: {} (MB)'.format(
                tensors, total
            )
        )
    return (tensors, total)


def process_memory_usage(to_print=True):
    process = psutil.Process()
    process_mem = process.memory_info().rss // 2**20 # in MB

    ram_info = psutil.virtual_memory()
    
    if to_print:
        print(f"Process memory usage: {process_mem} MB")  
    
        ram_info = psutil.virtual_memory()

    return process_mem, (ram_info.total - ram_info.available) // 2**20

    

# import time




class MPS_MemoryTracker(object):
    
    def __init__(self, clean_cache_before=True, clean_cache_after=False):
        self.clean_cache_before = clean_cache_before
        self.clean_cache_after = clean_cache_after
    
    def __enter__(self):
        if self.clean_cache_before:
            empty_cache()
        self.mps_mem_start = mps_memory_usage(to_print=False)
        self.mem_start = process_memory_usage(to_print=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.clean_cache_after:
            empty_cache()
        self.mps_mem_end = mps_memory_usage(to_print=False)
        self.mem_end = process_memory_usage(to_print=False)
        
        table = tabulate([
                ('Before', self.mps_mem_start[0], self.mps_mem_start[1], self.mem_start[0],self.mem_start[1], ),
                ('After', self.mps_mem_end[0], self.mps_mem_end[1], self.mem_end[0], self.mem_end[1], ),
                
                ('Diff', 
                '{0:+} MB'.format(self.mps_mem_end[0] - self.mps_mem_start[0]),
                '{0:+} MB'.format(self.mps_mem_end[1] - self.mps_mem_start[1]),
                '{0:+} MB'.format(self.mem_end[0] - self.mem_start[0]),
                '{0:+} MB'.format(self.mem_end[1] - self.mem_start[1]),
                )
            ],
            headers=['MPS tensors', 'MPS Total', 'Process Memory', 'Total System Memory']
        )
        print('######## Memory consumption ########:')
        print(table)
        