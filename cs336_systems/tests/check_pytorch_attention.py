import torch
from cs336_systems.modeling.naive_attention import NaiveAttention
from cs336_basics.model import scaled_dot_product_attention

f_attention = NaiveAttention.apply

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_queries = 128
num_keys = 128
dim = 64

q = torch.randn(batch_size, num_queries, dim, device=device)
k = torch.randn(batch_size, num_keys, dim, device=device)
v = torch.randn(batch_size, num_keys, dim, device=device)

# Our implementation
o = f_attention(q, k, v)

# Reference implementation
o_ref = scaled_dot_product_attention(q, k, v)

# Test closeness
is_close = torch.allclose(o, o_ref, rtol=1e-3, atol=1e-3)
print("NaiveAttention vs. reference allclose:", is_close)
if not is_close:
    print("Max abs diff:", (o - o_ref).abs().max().item())
    print("Mean abs diff:", (o - o_ref).abs().mean().item())
assert is_close, "NaiveAttention output does not match reference scaled_dot_product_attention"