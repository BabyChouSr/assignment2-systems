import submitit
import torch
from pathlib import Path
import pandas as pd

from cs336_systems.ddp.train_lm import DDPBenchmarkOutput, TrainLMConfig, train

configs = [
    TrainLMConfig(optimization="overlap_bucket", bucket_size_mb=1),
    TrainLMConfig(optimization="overlap_bucket", bucket_size_mb=10),
    TrainLMConfig(optimization="overlap_bucket", bucket_size_mb=100),
    TrainLMConfig(optimization="overlap_bucket", bucket_size_mb=1000),
]

def profile_model(config):
    # import pandas as pd

    # rows = []
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
    output = train(
        config
    )

    print(
        f"Output of benchmark: {output}"
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


    # if backend_type == "nccl":
    #     gpus_per_node = num_process
    # else:
    #     gpus_per_node = 0

    executor.update_parameters(
        timeout_min=5,
        slurm_partition="a2",
        slurm_qos="a2-qos",
        cpus_per_task=8,
        gpus_per_node=2,
        mem_gb=100,
        name=f"benchmarking_script",
        stderr_to_stdout=True,
    )

    for config in configs:
        job = executor.submit(
            profile_model,
            # TrainLMConfig(optimization="overlap_bucket"),
            config,
        )

    print(f"Submitted job ID: {job.job_id}")

    output = job.result()
    # print(output)
    # print(f"Finished job ID: {job.job_id}")