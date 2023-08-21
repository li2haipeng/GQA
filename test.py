import torch
import torch.nn.functional as F
import math
import time
# Optionally use the context manager to ensure one of the fused kerenels is run
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")

with torch.backends.cuda.sdp_kernel(enable_flash = False, enable_mem_efficient=True, enable_math=False):
    start = time.time()
    a = F.scaled_dot_product_attention(query,key,value, is_causal=False)
    print(time.time()-start)


# with torch.backends.cuda.sdp_kernel(enable_flash = False, enable_mem_efficient=False, enable_math=True):
#     start = time.time()
#     b = F.scaled_dot_product_attention(query,key,value, is_causal=False)
#     print(time.time()-start)

# Efficient implementation equivalent to the following:
# attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
# attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
start = time.time()
attn_weight = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(64)
attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query.dtype)
# attn_weight = torch.dropout(attn_weight, dropout_p)
b = torch.matmul(attn_weight, value)
print(time.time()-start)
print(a[0],b[0])
print(torch.equal(a,b))