import submitit
import torch
from pathlib import Path
import pandas as pd

from cs336_systems.benchmark.triton_bench import AttentionBenchmarkConfig, profile_attention
# from cs336_systems.defaults.model import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_2_7B

# TODO(chris): add back head dim 8?
head_dims = [16, 32, 64, 128]
seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
dtypes = [torch.bfloat16, torch.float32]

def profile_models(head_dims, seq_lens):
    import pandas as pd

    rows = []
    for attention_type in ["triton", None]:
        for head_dim in head_dims:
            for seq_len in seq_lens:
                for dtype in dtypes:
                    for pass_type in ["forward", "backward"]:
                        benchmark_output = profile_attention(
                            AttentionBenchmarkConfig(
                                head_dim=head_dim,
                                seq_len=seq_len,
                                pass_type=pass_type,
                                attention_type=attention_type,
                                dtype=dtype,
                            )
                        )
                        row = {
                            "head_dim": head_dim,
                            "seq_len": seq_len,
                            "pass": pass_type,
                            "dtype": str(dtype),
                            "attention_type": attention_type if attention_type is not None else "pytorch",
                            "mean": benchmark_output.mean_time,
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