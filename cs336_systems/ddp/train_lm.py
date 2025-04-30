import timeit
import os
import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import AdamW
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.cuda.nvtx as nvtx
from dataclasses import dataclass


# from cs336_systems.modeling.toy_model import ToyModel
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_systems.ddp.wrapper import DDP, DDPBucketed, DDPOptim

@dataclass
class TrainLMConfig:
    vocab_size: int = 10000
    context_length: int = 512
    d_model: int = 1600
    num_layers: int = 48
    num_heads: int = 25
    d_ff: int = 6400
    rope_theta: float = 10000

    train_path: str = "/data/c-cychou/tokenized/tinystories-train.npy"
    batch_size: int = 1
    seed: int = 42
    backend: str = "cuda"
    world_size: int = 2
    num_training_steps: int = 350

    # flatten_grads, overlap, None, overlap_bucket, "sharded_optim"
    optimization: str | None = None
    bucket_size_mb: int | None = None

@dataclass
class DDPBenchmarkOutput:
    mean_step_time: float
    mean_comm_time: float


# total_data_size = 128
# in_features = 10
# out_features = 1
# num_steps = 10
# world_size = 2
# backend = "cuda"
# seed = 42

def set_seed(seed, rank):
    process_seed = seed + rank

    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

    np.random.seed(process_seed)

    import random
    random.seed(process_seed)

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if backend == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def train_model(rank, config):
    print(config)
    set_seed(config.seed, rank)

    setup(rank, config.world_size, config.backend)
    device = torch.device(config.backend, rank)

    if config.backend == "cuda":
        torch.cuda.set_device(rank)

    # data_shard_size = data.size(0) // world_size
    # start_index = rank * data_shard_size
    # local_data = data[start_index:start_index + data_shard_size]
    # local_data = local_data.to(device)

    train_tokens = np.load(config.train_path, mmap_mode="r")

    # local_data = data[start_index:start_index + data_shard_size]
    
    model = BasicsTransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
        config.rope_theta,
    ).to(device)
    if config.optimization == "overlap":
        model = DDP(model)
    elif config.optimization == "overlap_bucket":
        model = DDPBucketed(model, config.bucket_size_mb)

    if config.optimization == "sharded_optim":
        optimizer = DDPOptim(model.parameters(), AdamW, lr=1e-5)
    else:
        optimizer = AdamW(model.parameters(), lr=1e-5)
    

    # if torch.cuda.is_available():

    print(f"After model initialization, memory allocated: {torch.cuda.max_memory_allocated(device=rank)}")

    training_step_times = []
    communication_times = []
    for step in range(config.num_training_steps):
        with nvtx.range("training step"):
            start_time = timeit.default_timer()

            x, y = get_batch(train_tokens, config.batch_size, config.context_length, str(device))
            # print(loss)
            # dist.barrier()
            with nvtx.range("forward pass"):
                logits = model(x)
            logits = logits.reshape(-1, config.vocab_size)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)

            with nvtx.range("backward pass"):
                loss.backward()

            start_comm_time = timeit.default_timer()

            with nvtx.range("communicating gradients"):
                if config.optimization == "flatten_grads":
                    all_gradients = []
                    for param in model.parameters():
                        if param.grad is not None:
                            all_gradients.append(param.grad)
                    flattened_grads = _flatten_dense_tensors(all_gradients)

                    if config.backend == "cuda":
                        dist.all_reduce(tensor=flattened_grads, op=dist.ReduceOp.AVG, async_op=False)
                    else:
                        dist.all_reduce(tensor=flattened_grads, async_op=False)
                    unflattened_grads = _unflatten_dense_tensors(flattened_grads, all_gradients)
                    for i, param in enumerate(model.parameters()):
                        if config.backend == "cuda":
                            param.grad = unflattened_grads[i]
                        else:
                            param.grad = unflattened_grads[i] / config.world_size
                elif config.optimization is not None and "overlap" in config.optimization:
                    model.finish_gradient_synchronization()
                else:
                    for param in model.parameters():
                        if config.backend == "cuda":
                            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
                        elif config.backend == "cpu":
                            dist.all_reduce(tensor=param.grad, async_op=False)
                            param.grad = param.grad / config.world_size
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_comm_time = timeit.default_timer()

            comm_duration = end_comm_time - start_comm_time
            communication_times.append(comm_duration)

            print(f"Right before optimizer step, memory allocated: {torch.cuda.max_memory_allocated(device=rank)}")
            with nvtx.range("optimizer step"):
                optimizer.step()
            print(f"Right after optimizer step, memory allocated: {torch.cuda.max_memory_allocated(device=rank)}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = timeit.default_timer()
            step_duration = end_time - start_time
            training_step_times.append(step_duration)

            print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, comm time = {comm_duration}, step time = {step_duration}", flush=True)
        
    
    mean_train_time = sum(training_step_times) / len(training_step_times)
    mean_comm_time = sum(communication_times) / len(communication_times)

    cleanup()

    print(DDPBenchmarkOutput(mean_step_time=mean_train_time, mean_comm_time=mean_comm_time))
    
@draccus.wrap()
def train(config: TrainLMConfig):
    mp.spawn(fn=train_model, args=(config, ), nprocs=config.world_size, join=True)

if __name__ == "__main__":
    train()