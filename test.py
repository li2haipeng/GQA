# -*- coding: utf-8 -*-
import torch
import time

start = time.time()
for i in range(10):
    a = torch.randint(high = 2,size=(8, 32, 512, 128), dtype=torch.float16).cuda()
    b = torch.randint(high = 2,size=(1, 32, 128, 512), dtype=torch.float16).cuda()
    # c = torch.stack([a,b], dim=2)
    z = torch.einsum("bhld, igdm->bhlm", a,b)
print(torch.cuda.memory_summary(0))
print(time.time()-start)








