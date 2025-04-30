import torch
import triton
import dataclasses
import draccus
import timeit
import numpy as np
from functools import partial
from cs336_systems.modeling.naive_attention import NaiveAttention
from cs336_systems.modeling.triton.attention import TritonAttention
from cs336_systems.benchmark.benchmarking_script import BenchmarkOutput

device = "cuda"
batch_size = 1
num_profile_steps = 100
seq_len = None

@dataclasses.dataclass
class AttentionBenchmarkConfig:
    head_dim: int
    seq_len: int
    pass_type: str
    dtype: torch.dtype
    attention_type: str | None = None

# @dataclasses.dataclass
# class AttentionBenchmarkOutput(BenchmarkOutput):
#     pre_backward_memory_usage: int

def _create_attn_inputs(config):
    requires_grad = True if config.pass_type == "backward" else False
    q = torch.randn(batch_size, config.seq_len, config.head_dim, requires_grad=requires_grad, dtype=config.dtype, device=device)
    k = torch.randn(batch_size, config.seq_len, config.head_dim, requires_grad=requires_grad, dtype=config.dtype, device=device)
    v = torch.randn(batch_size, config.seq_len, config.head_dim, requires_grad=requires_grad, dtype=config.dtype, device=device)
    return q, k, v

def forward(config, attention_func):
    q, k, v = _create_attn_inputs(config)
    def forward_pass():
        attention_func(q, k, v)
        torch.cuda.synchronize()

    return forward_pass


def backward(config, attention_func):
    q, k, v = _create_attn_inputs(config)
    def backward_pass():
        output = attention_func(q, k, v)
        output.mean().backward()
        torch.cuda.synchronize()
    
    return backward_pass

def create_triton_attention_forward_pass(config):
    def run_forward_pass(q, k, v):
        with torch.autocast(device, dtype=config.dtype):
            outputs = TritonAttention.apply(q, k, v, True)

        return outputs
    
    return run_forward_pass
        

def create_pytorch_attention_forward_pass(config):
    def run_forward_pass(q, k, v):
        # mask = torch.arange(config.seq_len, device=device)[None, :, None] >= torch.arange(config.seq_len, device=device)[None, None, :]
        with torch.autocast(device, dtype=config.dtype):
            # outputs = scaled_dot_product_attention(q, k, v, mask)
            outputs = NaiveAttention.apply(q, k, v, True)
    
        return outputs
    
    return run_forward_pass

@draccus.wrap()
def profile_attention(config: AttentionBenchmarkConfig):
    if config.pass_type == "forward":
        profile_func = forward
    elif config.pass_type == "backward":
        profile_func = backward

    if config.attention_type == "triton":
        attention_func = create_triton_attention_forward_pass(config)
    else:
        attention_func = create_pytorch_attention_forward_pass(config)
        
    # requires_grad = True if config.pass_type == "backward" else False

    # global seq_len
    # seq_len = config.seq_len

    # for _ in range(10):
    #     profile_func(q, k, v, attention_func)

    times = []
    # memory_usages = []
    # torch.cuda.synchronize()
    # for _ in range(num_profile_steps):
        # time, memory_usage = profile_func(q, k, v, attention_func)
        # time = profile_func(config, attention_func)
    time = triton.testing.do_bench(profile_func(config, attention_func))
        # times.append(time)
        # memory_usages.append(memory_usage)

    # mean_time = sum(times) / len(times)
    # std_dev = np.std(times, ddof=1)
    # mean_memory_usage = sum(memory_usages) / len(memory_usages)

    return BenchmarkOutput(mean_time=time, std_dev=0.00)
    