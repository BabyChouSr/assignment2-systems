#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --job-name=benchmarking_script
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/c-cychou/assignment2-systems/slurm_outputs/%j_0_log.out
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --signal=USR2@90
#SBATCH --time=5

# MODEL_TYPES=("none" "overlap")
# MODEL_TYPES=("sharded_optim")
MODEL_TYPES=("overlap")
BUCKET_SIZES=(10 100 1000)

for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
    for BUCKET_SIZE in "${BUCKET_SIZES[@]}"
    do
        # Create the output directory if it doesn't exist
        mkdir -p nvtx_results/ddp_bucket
        OUTPUT_FILE="nvtx_results/ddp/result_${MODEL_TYPE}_${BUCKET_SIZE}"
        uv run nsys profile --force-overwrite true -o $OUTPUT_FILE python cs336_systems/ddp/train_lm.py --config cs336_systems/defaults/ddp/${MODEL_TYPE}.yaml --optimization overlap_bucket --bucket_size_mb ${BUCKET_SIZE}
    done
done
