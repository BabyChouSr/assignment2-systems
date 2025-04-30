import torch
import dataclasses
import draccus
import timeit
import numpy as np
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.benchmark.benchmarking_script import BenchmarkOutput

device = "cuda"
batch_size = 8
num_profile_steps = 100
seq_len = None

@dataclasses.dataclass
class AttentionBenchmarkConfig:
    head_dim: int
    seq_len: int
    pass_type: str
    attention_type: str | None = None

@dataclasses.dataclass
class AttentionBenchmarkOutput(BenchmarkOutput):
    pre_backward_memory_usage: int

def forward_pass(q, k, v, attention_func):
    start_time = timeit.default_timer()
    attention_func(q, k, v)
    torch.cuda.synchronize()
    end_time = timeit.default_timer()

    return end_time - start_time, 0

def backward_pass(q, k, v, attention_func):
    output = attention_func(q, k, v)
    mem_before = torch.cuda.memory_allocated(device)
    torch.cuda.synchronize()
    start_time = timeit.default_timer()
    output.mean().backward()
    torch.cuda.synchronize()
    end_time = timeit.default_timer()

    return end_time - start_time, mem_before

@draccus.wrap()
def profile_attention(config: AttentionBenchmarkConfig):
    if config.pass_type == "forward":
        profile_func = forward_pass
    elif config.pass_type == "backward":
        profile_func = backward_pass

    if config.attention_type == "compiled":
        attention_func = torch.compile(scaled_dot_product_attention)
    else:
        attention_func = scaled_dot_product_attention
        
    requires_grad = True if config.pass_type == "backward" else False

    global seq_len
    seq_len = config.seq_len

    q = torch.randn(batch_size, config.seq_len, config.head_dim, requires_grad=requires_grad).to(device)
    k = torch.randn(batch_size, config.seq_len, config.head_dim, requires_grad=requires_grad).to(device)
    v = torch.randn(batch_size, config.seq_len, config.head_dim, requires_grad=requires_grad).to(device)

    for _ in range(10):
        profile_func(q, k, v, attention_func)

    times = []
    memory_usages = []
    torch.cuda.synchronize()
    for _ in range(num_profile_steps):
        time, memory_usage = profile_func(q, k, v, attention_func)
        times.append(time)
        memory_usages.append(memory_usage)

    mean_time = sum(times) / len(times)
    std_dev = np.std(times, ddof=1)
    mean_memory_usage = sum(memory_usages) / len(memory_usages)

    return AttentionBenchmarkOutput(mean_time=mean_time, std_dev=std_dev, pre_backward_memory_usage=mean_memory_usage)
    