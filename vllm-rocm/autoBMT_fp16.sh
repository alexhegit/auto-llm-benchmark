#!/bin/bash

# Define an array of model paths
MODEL_PATHS=(
    "/data/llm/Meta-Llama-3.1-8B"
    "/data/llm/Meta-Llama-3.1-70B"
    "/data/llm/Meta-Llama-3.1-405B"
)

# Create a directory to save output files
OUTPUT_DIR="./OUTPUT/BMT"
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
        echo "Starting test for model path: $MODEL_PATH"
        torchrun --standalone --nproc_per_node=8 /app/vllm/benchmarks/benchmark_throughput.py \
            --model "$MODEL_PATH" \
            --dtype float16 \
            --max-num-batched-tokens 65536 \
            --gpu-memory-utilization 0.99 \
            --max-model-len 8192 \
            --num-prompts 2000 \
            --tensor-parallel-size 8 \
            --input-len 512 \
            --output-len 2048 \
            --output-json "$OUTPUT_FILE"
        echo "Completed test for model path: $MODEL_PATH, results saved to $OUTPUT_FILE"
	echo `date`
    } &> "$LOG_FILE"
    
    echo "Log saved to $LOG_FILE"
done
