import os
import timeit
from dataclasses import dataclass

import draccus

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

@dataclass
class BenchmarkCommunicationConfig:
    backend_type: str
    data_size_mb: int
    num_processes: int

def setup(rank, world_size, backend_type):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend_type, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_distributed_comm(rank, world_size, data_size_mb, backend_type):
    setup(rank, world_size, backend_type)

    if backend_type == "gloo":
        device_type = "cpu"
    else:
        device_type = "cuda"

    device = torch.device(device_type, rank)
    
    num_elements = data_size_mb * 1024 * 1024 // 4
    data = torch.randn(num_elements, device=device, dtype=torch.float32)
    for _ in range(5):
        dist.all_reduce(data, async_op=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(10):
        start_time = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = timeit.default_timer() - start_time
        times.append(duration)

    sent_bytes = data.element_size() * data.numel() * 2 * (world_size - 1)
    total_duration = world_size * sum(times) / len(times)
    bandwidth = sent_bytes / total_duration

    print(f"[all_reduce] Rank {rank}: Data Size: {data_size_mb}MB: Backend: {backend_type} all_reduce measured bandwidth = {bandwidth / 1024**3} GB/s", flush=True)
    cleanup()

@draccus.wrap()
def benchmark_comms(config: BenchmarkCommunicationConfig):
    world_size = 4
    mp.spawn(fn=run_distributed_comm, args=(config.num_processes, config.data_size_mb, config.backend_type), nprocs=config.num_processes, join=True)

if __name__ == "__main__":
    benchmark_comms()
