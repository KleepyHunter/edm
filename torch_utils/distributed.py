# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
from . import training_stats
import torch.distributed as dist

#----------------------------------------------------------------------------

def init():
    # Set default environment variables if not set
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')

    # Set backend based on availability
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    
    # Initialize process group
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    
    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    
    # Sync device if running with multiple GPUs
    sync_device = torch.device('cuda') if torch.distributed.get_world_size() > 1 and torch.cuda.is_available() else None

    # Initialize training stats with multiprocessing sync
    training_stats.init_multiprocessing(rank=dist.get_rank(), sync_device=sync_device)

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def should_stop():
    return False

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
