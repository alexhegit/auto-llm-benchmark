#!/bin/bash

# Define an array of model paths
_MODEL_PATHS=(
    "/data/llm/Meta-Llama-3.1-70B-Instruct-FP8-KV"
    "/data/llm/Meta-Llama-3.1-405B-Instruct-FP8-KV"
)
MODEL_PATHS=(
    "/data/llm/Meta-Llama-3.1-70B-Instruct-FP8-KV"
)

QFILE="/quantized/llama.safetensors"

# Create a directory to save output files
OUTPUT_DIR="./OUTPUT/BMT_TP8_FP8"
mkdir -p "$OUTPUT_DIR"

# Loop through each model path and run the test
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    # Extract model name for output file naming
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}.json"
    LOG_FILE="${OUTPUT_DIR}/${MODEL_NAME}.log"

    echo "Testing model path: $MODEL_PATH"
    # Redirect output to log file
    {
        echo `date`
        QPATH=${MODEL_PATH}/${QFILE}
        echo "Starting test for model path: $MODEL_PATH"
        echo "Quantized weights path: $QPATH"
        # Set benchmark options by refering to https://github.com/powderluv/vllm-docs
        torchrun --standalone --nproc_per_node=8 /app/vllm/benchmarks/benchmark_throughput.py \
            --model "$MODEL_PATH" \
            --quantization fp8 --kv-cache-dtype fp8 \
            --dtype half \
            --max-num-batched-tokens 65536 \
            --gpu-memory-utilization 0.99 \
            --max-model-len 8192 \
            --num-prompts 2000 \
            --distributed-executor-backend mp \
            --num-scheduler-steps 10 \
            --swapspace 16 \
            --tensor-parallel-size 8 \
            --input-len 512 \
            --output-len 2048 \
            --output-json "$OUTPUT_FILE"
        echo "Completed test for model path: $MODEL_PATH, results saved to $OUTPUT_FILE"
        echo `date`
    } &> "$LOG_FILE"

    echo "Log saved to $LOG_FILE"
done
