import submitit
import torch
from pathlib import Path
import pandas as pd

from cs336_systems.benchmark.benchmarking_script import benchmark, BenchmarkConfig
from cs336_systems.defaults.model import gpt2_small, gpt2_medium, gpt2_large, gpt2_xl, gpt2_2_7B

models = [gpt2_2_7B]
context_lengths = [512]
dtypes = [torch.float32]

def profile_models(models):
    import pandas as pd

    rows = []
    for model_config in models:
        for context_length in context_lengths:
            for dtype in dtypes:
                for pass_type in ["forward", "all"]:
                    model_config.context_length = context_length
                    model_config.profile_pass = pass_type
                    model_config.dtype = dtype
                    model_config.output_file = f"memory_results/{pass_type}-{context_length}-{str(dtype).split('.')[1]}.pkl"
                    
                    benchmark_output = benchmark(model_config)
                    row = {
                        "d_model": model_config.d_model,
                        "d_ff": model_config.d_ff,
                        "n_layers": model_config.num_layers,
                        "n_heads": model_config.num_heads,
                        "vocab": model_config.vocab_size,
                        "seq_len": model_config.context_length,
                        "theta": model_config.rope_theta,
                        "w": model_config.warmup_steps,
                        "n": model_config.profile_steps,
                        "dtype": model_config.dtype,
                        "pass": pass_type,
                        "mean": benchmark_output.mean_time,
                        "std": benchmark_output.std_dev,
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_latex(index=False, float_format="{:.2f}".format))
    df.to_csv("results/benchmarking_script_gpt2.csv")
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
            models,
    )
    print(f"Submitted job ID: {job.job_id}")