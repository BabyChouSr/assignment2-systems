import draccus
from dataclasses import dataclass
import torch
import torch.cuda.nvtx as nvtx
import timeit
import numpy as np

import cs336_basics
from cs336_systems.modeling.attention import annotated_scaled_dot_product_attention
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

@dataclass
class BenchmarkConfig:
    # Model initialization hyperparameters
    num_layers: int
    vocab_size: int
    context_length: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: int

    warmup_steps: int
    profile_steps: int
    profile_pass: str
    batch_size: int = 4
    dtype: torch.dtype = torch.float32

@dataclass
class BenchmarkOutput:
    mean_time: int
    std_dev: int

def run_forward_pass(model, batch):
    start_time = timeit.default_timer()
    model(batch)
    end_time = timeit.default_timer()
    torch.cuda.synchronize()
    return end_time - start_time

def run_backward_pass(model, batch):
    start_time = timeit.default_timer()
    loss = cross_entropy(model(batch), batch)
    loss.backward()
    torch.cuda.synchronize()
    end_time = timeit.default_timer()

    return end_time - start_time

def run_all_pass_nvtx(model, batch):
    optimizer = AdamW(model.parameters())
    start_time = timeit.default_timer()
    with nvtx.range("forward pass"):
        output = model(batch)

    with nvtx.range("backward pass"):
        loss = cross_entropy(model(batch), batch)
        loss.backward()
    
    with nvtx.range("optimizer step"):
        optimizer.step()

    torch.cuda.synchronize()
    end_time = timeit.default_timer()
    return end_time - start_time

def apply_monkey_patch():
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

@draccus.wrap()
def benchmark(config: BenchmarkConfig):
    apply_monkey_patch()

    model = BasicsTransformerLM(
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    ).to(device)

    batch = torch.randint(0, config.vocab_size, (4, config.context_length, ), dtype=torch.long, device=device)

    if config.profile_pass == "forward":
        profile_func = run_forward_pass
    elif config.profile_pass == "backward":
        profile_func = run_backward_pass
    elif config.profile_pass == "all":
        profile_func = run_all_pass_nvtx

    for _ in range(config.warmup_steps):
        profile_func(model, batch)

    torch.cuda.synchronize()

    times = []
    for _ in range(config.profile_steps):
        with torch.autocast(device, dtype=config.dtype):
            times.append(profile_func(model, batch))

    mean_time = sum(times) / len(times)
    std_dev = np.std(times, ddof=1)

    return BenchmarkOutput(mean_time=mean_time, std_dev=std_dev)

if __name__ == "__main__":
    benchmark()