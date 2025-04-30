import timeit
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW

from cs336_systems.modeling.toy_model import ToyModel


total_data_size = 128
in_features = 10
out_features = 1
num_steps = 10
world_size = 2
backend = "cuda"
seed = 42

def set_seed():
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if backend == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def train_model(rank, world_size, data):
    set_seed()

    setup(rank, world_size)
    device = torch.device(backend, rank)

    data_shard_size = data.size(0) // world_size
    start_index = rank * data_shard_size
    local_data = data[start_index:start_index + data_shard_size]
    local_data = local_data.to(device)
    model = ToyModel(in_features, out_features).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    # if torch.cuda.is_available():

    training_step_times = []
    communication_times = []
    for step in range(num_steps):
        start_time = timeit.default_timer()
        output = model(local_data)
        loss = output.square().mean()
        # print(loss)
        # dist.barrier()
        loss.backward()

        start_comm_time = timeit.default_timer()
        for param in model.parameters():
            if backend == "cuda":
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
            elif backend == "cpu":
                dist.all_reduce(tensor=param.grad, async_op=False)
                param.grad = param.grad / world_size
        end_comm_time = timeit.default_timer()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        communication_times.append(end_comm_time - start_comm_time)

        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = timeit.default_timer()
        training_step_times.append(end_time - start_time)

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}", flush=True)
        
    mean_train_time = sum(training_step_times) / len(training_step_times)
    mean_comm_time = sum(communication_times) / len(communication_times)
    

def train():
    data = torch.randn(total_data_size, in_features)
    mp.spawn(fn=train_model, args=(world_size, data), nprocs=world_size, join=True)

if __name__ == "__main__":
    train()