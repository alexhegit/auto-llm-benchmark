#!/bin/bash

# Define combinations of input-len and output-len
_COMBINATIONS=(
    "128 128"
    "128 2048"
    "2048 2048"
    "2048 128"
)

COMBINATIONS=(
    "2048 2048"
    "2048 128"
)

# Define batch sizes
BATCH_SIZES=(
    1 2 4 8 16 32 64 128 256
)

# Create directories to save output files
OUTPUT_DIR="./OUTPUT"
BML_DIR="${OUTPUT_DIR}/BML_FP8"
mkdir -p "$BML_DIR"

# Loop through each combination of input-len, output-len, and batch-size
for COMBINATION in "${COMBINATIONS[@]}"; do
    INPUT_LEN=$(echo $COMBINATION | cut -d ' ' -f 1)
    OUTPUT_LEN=$(echo $COMBINATION | cut -d ' ' -f 2)

    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        LOG_FILE="${BML_DIR}/Meta-Llama-3.1-70B_i${INPUT_LEN}_o${OUTPUT_LEN}_b${BATCH_SIZE}.log"
        JSON_FILE="${BML_DIR}/Meta-Llama-3.1-70B_i${INPUT_LEN}_o${OUTPUT_LEN}_b${BATCH_SIZE}.json"

        echo "Testing with input-len: $INPUT_LEN, output-len: $OUTPUT_LEN, and batch-size: $BATCH_SIZE"
        # Define the command with the current combination
        COMMAND="python3 /app/vllm/benchmarks/benchmark_latency.py \
            --tensor-parallel-size 8 \
            --num-iters-warmup 5 \
            --num-iters 10 \
            --worker-use-ray \
            --model /data/llm/Meta-Llama-3.1-70B \
            --quantized-weights-path /quantized/llama.safetensors \
            --quantization fp8 \
            --kv-cache-dtype fp8 \
            --dtype float16 \
            --input-len $INPUT_LEN \
            --output-len $OUTPUT_LEN \
            --batch-size $BATCH_SIZE \
            --output-json $JSON_FILE"

        # Redirect output to log file
        {
            echo `date`
            echo "Running command: $COMMAND"
            $COMMAND
            echo "Completed test with input-len: $INPUT_LEN, output-len: $OUTPUT_LEN, and batch-size: $BATCH_SIZE"
            echo `date`
        } &> "$LOG_FILE"

        echo "Log saved to $LOG_FILE"
    done
done

