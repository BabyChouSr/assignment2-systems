import submitit
import torch
from pathlib import Path
import pandas as pd

# from cs336_systems.benchmark.triton_bench import AttentionBenchmarkConfig, profile_attention
# from cs336_systems.defaults.model import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_2_7B
from cs336_systems.benchmark.benchmark_comm import BenchmarkCommunicationConfig, benchmark_comms

# TODO(chris): add back head dim 8?
# head_dims = [16, 32, 64, 128]
# seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
# dtypes = [torch.bfloat16, torch.float32]

backend_types = ["gloo", "nccl"]
data_sizes = [1, 10, 100, 1000]
num_processes = [2, 4, 6]

# backend_types = ["gloo"]
# data_sizes = [1]
# num_processes = [2]

def profile_models(config):
    import pandas as pd

    rows = []
    # for backend_type in backend_types:
    #     for data_size in data_sizes:
    #         for num_process in num_processes:
                # benchmark_output = profile_attention(
                #     AttentionBenchmarkConfig(
                #         head_dim=head_dim,
                #         seq_len=seq_len,
                #         pass_type=pass_type,
                #         attention_type=attention_type,
                #         dtype=dtype,
                #     )
                # )
    benchmark_comms(
        config
    )
                # row = {
                #     "head_dim": head_dim,
                #     "seq_len": seq_len,
                #     "pass": pass_type,
                #     "dtype": str(dtype),
                #     "attention_type": attention_type if attention_type is not None else "pytorch",
                #     "mean": benchmark_output.mean_time,
                # }
                # rows.append(row)

    # df = pd.DataFrame(rows)
    # print(df.to_latex(index=False, float_format="{:.5f}".format))
    # df.to_csv("results/attention_script_gpt2.csv")
    # return df


if __name__ == "__main__":
    folder = Path("slurm_outputs")
    folder.mkdir(exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)

    for backend_type in backend_types:
        for data_size in data_sizes:
            for num_process in num_processes:
                config = BenchmarkCommunicationConfig(
                    backend_type=backend_type,
                    data_size_mb=data_size,
                    num_processes=num_process,
                )


                if backend_type == "nccl":
                    gpus_per_node = num_process
                else:
                    gpus_per_node = 0

                executor.update_parameters(
                    timeout_min=5,
                    slurm_partition="a2",
                    slurm_qos="a2-qos",
                    cpus_per_task=8,
                    gpus_per_node=gpus_per_node,
                    mem_gb=100,
                    name=f"benchmarking_script",
                    stderr_to_stdout=True,
                )

                job = executor.submit(
                        profile_models, 
                        config,
                )

                print(f"Submitted job ID: {job.job_id}")

                output = job.result()
                print(f"Finished job ID: {job.job_id}")