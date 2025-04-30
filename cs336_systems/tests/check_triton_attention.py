import torch
from cs336_systems.modeling.triton.attention import TritonAttention
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.modeling.naive_attention import NaiveAttention

f_attention = TritonAttention.apply
f_ref = scaled_dot_product_attention
# f_ref = NaiveAttention.apply

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_queries = 16
num_keys = 16
dim = 256


def test_triton_attention():
    q = torch.randn(batch_size, num_queries, dim, device=device)
    k = torch.randn(batch_size, num_keys, dim, device=device)
    v = torch.randn(batch_size, num_keys, dim, device=device)

    # Our implementation
    o = f_attention(q, k, v)

    # Reference implementation
    o_ref = f_ref(q, k, v)

    # Test closeness
    is_close = torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2)
    print("NaiveAttention vs. reference allclose:", is_close)
    if not is_close:
        print("Max abs diff:", (o - o_ref).abs().max().item())
        print("Mean abs diff:", (o - o_ref).abs().mean().item())
    assert is_close, "NaiveAttention output does not match reference scaled_dot_product_attention"

# maybe_ls = [t for t in o.grad_fn.saved_tensors if t.shape == (q.shape[0], q.shape[1])]
# ls = maybe_ls[0]

# qk = torch.einsum("... qd, ... kd->... qk", q, k)
# ref_ls = torch.logsumexp(qk, dim=-1)
# # Test closeness
# is_close = torch.allclose(ref_ls, ls, rtol=1e-3, atol=1e-3)
# print("NaiveAttention vs. reference allclose:", is_close)
# if not is_close:
#     print("Max abs diff:", (o - o_ref).abs().max().item())
#     print("Mean abs diff:", (o - o_ref).abs().mean().item())
# assert is_close, "NaiveAttention LSE does not match reference scaled_dot_product_attention"
