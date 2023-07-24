# -*- coding: utf-8 -*-
import torch
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")

print(torch.cuda.current_device())



