#!/bin/bash

MODEL_TYPES=("gpt2_small" "gpt2_medium" "gpt2_large" "gpt2_xl" "gpt2_2_7b")

for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
    # Create the output directory if it doesn't exist
    mkdir -p nvtx_results/${MODEL_TYPE}

    for CONTEXT_LENGTH in 128 256 512 1024
    do
        OUTPUT_FILE="nvtx_results/${MODEL_TYPE}/result_${MODEL_TYPE}_${CONTEXT_LENGTH}"
        uv run nsys profile --force-overwrite true -o $OUTPUT_FILE python cs336_systems/benchmark/benchmarking_script.py --config cs336_systems/defaults/${MODEL_TYPE}.yaml --context_length $CONTEXT_LENGTH
    done
done
