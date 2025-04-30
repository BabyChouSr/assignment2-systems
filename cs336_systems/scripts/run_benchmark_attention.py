import submitit
import torch
from pathlib import Path
import pandas as pd

from cs336_systems.benchmark.benchmark_attention import AttentionBenchmarkConfig, profile_attention
from cs336_systems.defaults.model import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_2_7B

head_dims = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192, 16384]

def profile_models(head_dims, seq_lens):
    import pandas as pd

    rows = []
    for head_dim in head_dims:
        for seq_len in seq_lens:
            for pass_type in ["forward", "backward"]:
                benchmark_output = profile_attention(
                    AttentionBenchmarkConfig(
                        head_dim=head_dim,
                        seq_len=seq_len,
                        pass_type=pass_type,
                        attention_type="compiled",
                    )
                )
                row = {
                    "head_dim": head_dim,
                    "seq_len": seq_len,
                    "pass": pass_type,
                    "mean": benchmark_output.mean_time,
                    "std": benchmark_output.std_dev,
                    "mean_memory": benchmark_output.pre_backward_memory_usage,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_latex(index=False, float_format="{:.5f}".format))
    df.to_csv("results/attention_script_gpt2.csv")
    return df


if __name__ == "__main__":
    folder = Path("slurm_outputs")
    folder.mkdir(exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(
        timeout_min=120,
        slurm_partition="a1-batch",
        slurm_qos="a1-batch-qos",
        cpus_per_task=8,
        gpus_per_node=1,
        mem_gb=100,
        name=f"benchmarking_script",
        stderr_to_stdout=True,
    )
    job = executor.submit(
            profile_models, 
            head_dims,
            seq_lens,
    )
    print(f"Submitted job ID: {job.job_id}")