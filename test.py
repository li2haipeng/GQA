# -*- coding: utf-8 -*-
import torch
import time
a = torch.randint(high = 2,size=(512, 32, 1, 128), dtype=torch.float16).cuda(0)
b = torch.randint(high = 2,size=(32, 2500, 128), dtype=torch.float16).cuda(0)
c = torch.randint(high = 2,size=(512, 32, 256, 128), dtype=torch.float16).cuda(0)
start = time.time()
# for i in range(5):

#     # c = torch.stack([a,b], dim=2)
#     # z = torch.einsum("bhld, igdm->bhlm", a,b)
#     t0 = time.time()
#     z = torch.matmul(a,b.transpose(-2,-1))
#     print("1",time.time()-t0)
#     zz = torch.matmul(a, c.transpose(2,3))
#     print("2",time.time()-t0)
#     r = torch.cat([z, zz], dim=-1) 
#     print(r.shape)
# print(torch.cuda.memory_summary(0))
# print(time.time()-start)


# start = time.time()

for i in range(5):
    a = torch.randint(high = 2,size=(512, 32, 1, 128), dtype=torch.float16).cuda(0)
    b = torch.randint(high = 2,size=(512, 32, 2756, 128), dtype=torch.float16).cuda(0)

    # z = torch.einsum("bhld, igdm->bhlm", a,b)
    z = torch.matmul(a,b.transpose(-2,-1))
    print(z.shape)
    # zz = torch.matmul(a, c.transpose(2,3))
print(torch.cuda.memory_summary(0))
print(time.time()-start)






