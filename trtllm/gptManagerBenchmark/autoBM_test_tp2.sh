#!/bin/bash

usage() {
    echo "Usage: $0 --engine <engine_path> --dataset <dataset_path>"
    exit 1
}

# Parse required arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --engine) ENGINE_PATH="$2"; shift 2 ;;
        --dataset) DATASET_PATH="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
done

# Check if required arguments are provided
if [ -z "$ENGINE_PATH" ] || [ -z "$DATASET_PATH" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Define the combinations isq-len and osq-len
COMBINATIONS=(
    "100 1000"
    "1000 1000"
    "10000 1000"
    "50000 1000"
    "100000 1000"
)
_COMBINATIONS=(
    "50000 1000"
)

# Define batch sizes
BATCH_SIZES=(
    1 10 100
)

WROOT="/ws/m1"
TP=2
BM_RESULT=${WROOT}/TBM/${MODEL_NAME}

# Define model name
#MODEL_NAME="Meta-Llama-3.1-70B"
#ENGINE=/myworkspace/trtllm_engine_trtllm_llama3.1_70b_gpus8_8_fp8
#DATASET_PATH="./dataset"
MODEL_NAME=$(basename "$ENGINE_PATH")

# Create directories to save output files
BM_RESULT=${WROOT}/TBM_R/${MODEL_NAME}
mkdir -p "$BM_RESULT"

# Function to run the benchmark
run_benchmark() {
    local INPUT_LEN=$1
    local OUTPUT_LEN=$2
    local BATCH_SIZE=$3
    local NUM_SAMPLES=$((BATCH_SIZE * 1))
    #local DATASET="${DATASET_PATH}/${MODEL_NAME}-Instruct_tokens-fixed-lengths_${INPUT_LEN}_${OUTPUT_LEN}.json"
    local DATASET="${DATASET_PATH}/${MODEL_NAME}_${INPUT_LEN}_${OUTPUT_LEN}"
    local LOG_FILE="${BM_RESULT}/${MODEL_NAME}_i${INPUT_LEN}_o${OUTPUT_LEN}_b${BATCH_SIZE}.log"
    local CSV_FILE="${BM_RESULT}/${MODEL_NAME}_i${INPUT_LEN}_o${OUTPUT_LEN}_b${BATCH_SIZE}.csv"

    echo "TRTLLM Benchmark: $MODEL_NAME with input-len: $INPUT_LEN, output-len: $OUTPUT_LEN, and batch-size: $BATCH_SIZE"
    # Define the command with the current combination
    COMMAND="mpirun -n $TP --allow-run-as-root --oversubscribe /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
   	--engine_dir $ENGINE_PATH \
	--request_rate -1 \
	--streaming true \
        --warm_up 10 \
    	--max_num_samples $NUM_SAMPLES \
	--output_csv $CSV_FILE\
        --static_emulated_batch_size $BATCH_SIZE \
        --static_emulated_timeout 100 \
	--kv_cache_free_gpu_mem_fraction 0.80 \
        --dataset $DATASET"

    # Debugging: Print the command to be executed
    echo "Running command: $COMMAND"

    # Redirect output to log file
    {
        echo `date`
        echo "Running command: $COMMAND"
        if eval $COMMAND; then
            echo "Completed test with input-len: $INPUT_LEN, output-len: $OUTPUT_LEN, and batch-size: $BATCH_SIZE"
        else
            echo "Error occurred during test with input-len: $INPUT_LEN, output-len: $OUTPUT_LEN, and batch-size: $BATCH_SIZE"
        fi
        echo `date`
    } &> "$LOG_FILE"

    echo "Log saved to $LOG_FILE"
}

# Loop through each combination of input-len, output-len, and batch-size
for COMBINATION in "${COMBINATIONS[@]}"; do
    INPUT_LEN=$(echo $COMBINATION | cut -d ' ' -f 1)
    OUTPUT_LEN=$(echo $COMBINATION | cut -d ' ' -f 2)

    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        run_benchmark "$INPUT_LEN" "$OUTPUT_LEN" "$BATCH_SIZE"
    done
done
