#!/bin/bash

# Function to create a directory if it doesn't exist
mk_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then    
        echo "Creating new folder '$dir'"
        mkdir -p "$dir"
    fi 
}

# Function to clean a directory (not delete the directory itself)
clean_dir() {
    local dir="$1"
    if [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "Cleaning up folder '$dir'"
        rm -rf "$dir"/*
    fi 
}

# Function to show usage (if undefined, remove this line)
usage() {
    echo "Usage: $0 --model_dir <DIR> --data_type <TYPE> --tp <TP_SIZE>"
    exit 1
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_dir) MODEL_DIR="$2"; shift ;;
        --data_type) DATA_TYPE="$2"; shift ;;
        --tp) TP_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$MODEL_DIR" || -z "$DATA_TYPE" || -z "$TP_SIZE" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

WROOT="/ws/m1"
MODEL_NAME=$(basename "$MODEL_DIR")

echo "Working Root DIR: $WROOT"
echo "Convert the Model from: $MODEL_DIR"

CKPT_ROOT="$WROOT/trtllm_ckpt"
ENG_ROOT="$WROOT/trtllm_engines"

# Create necessary directories
mk_dir "$CKPT_ROOT"
mk_dir "$ENG_ROOT"

CKPT_DIR="$CKPT_ROOT/$MODEL_NAME"
ENGINE_DIR="$ENG_ROOT/$MODEL_NAME"

CONVERT_CLI="python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py"

# Function to convert the checkpoint
convert_ckpt() { 
    # weights convert: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama
    #local COMMAND="$CONVERT_CLI --model_dir $MODEL_DIR --dtype $DATA_TYPE --output_dir $CKPT_DIR --tp_size $TP_SIZE"
    local COMMAND="python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir \"$MODEL_DIR\" --dtype \"$DATA_TYPE\" --output_dir \"$CKPT_DIR\" --tp_size \"$TP_SIZE\""

    echo "RUN: $COMMAND"

    START=$(date +%s)

    if eval "$COMMAND"; then
        echo "Done: checkpoint saved in $CKPT_DIR"
    else
        echo "Error: convert failed"
        exit 1
    fi

    END=$(date +%s)
    ETIME=$((END - START))
    echo -e "\n    Took $ETIME seconds"
}

#convert_ckpt


#max_num_tokens=4096
#max_batch_size=2048

# Function to build the engine
# https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html

# Set args of trtllm-build for the 2nd bigest test case: isl=100000, osl=1000, bs=10
# Test failed with OOM 
#MAX_NUM_TOKENS=1000000 
#MAX_NUM_TOKENS=1000000 

# Try for isl=100000, osl=1000, bs=1,10,100
# Llama-3.1-70B, work fine with bs=1,10,100
#MAX_NUM_TOKENS=100000  # build failed about OOM
MAX_NUM_TOKENS=4096
#MAX_INPUT_LEN=10000
MAX_SEQ_LEN=10000
#MAX_BATCH_SIZE=1


build_engine() {
	echo "RUN: trtllm-build checkpoint is ${CKPT_DIR}"
	trtllm-build --checkpoint_dir $CKPT_DIR --use_fused_mlp enable \
        --gpt_attention_plugin $DATA_TYPE \
        --output_dir $ENGINE_DIR \
        --workers $TP_SIZE \
        --max_num_tokens $MAX_NUM_TOKENS \
	--max_seq_len $MAX_SEQ_LEN \
        --use_paged_context_fmha enable
}

build_engine 
