#!/bin/bash

WROOT="/ws/m1"

mk_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then    
        echo "create new folder '$dir' "
        mkdir -p "$dir"
    fi 
}

clean_dir() {
    local dir="$1"
    if [ "$(ls -A "$dir")" ]; then
        echo " Clean up folder '$dir' "
        rm -rf "$dir"
    fi 
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_dir) MODEL_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --requests) REQUESTS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --prompt_len) PROMPT_LEN="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
    case $1 in
        --new_tokens) NEW_TOKENS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
done

REQUESTS_SP=""
for i in $(echo $REQUESTS | tr "," "\n")
do
  REQUESTS_SP="$REQUESTS_SP $i"
done

PROMPT_LEN_SP=""
for i in $(echo $PROMPT_LEN | tr "," "\n")
do
  PROMPT_LEN_SP="$PROMPT_LEN_SP $i"
done

NEW_TOKENS_SP=""
for i in $(echo $NEW_TOKENS | tr "," "\n")
do
  NEW_TOKENS_SP="$NEW_TOKENS_SP $i"
done

WROOT="/ws/m1"
#MODEL_DIR="/ws/m1/Llama-3.1-70B/"
MODEL_NAME=$(basename "$MODEL_DIR")

hf_model_path=$MODEL_DIR

echo "=hyper params start="
echo "model: $MODEL_NAME"
echo "requests: $REQUESTS_SP"
echo "input len: $PROMPT_LEN_SP"
echo "output len: $NEW_TOKENS_SP"

requests=$REQUESTS_SP
isl=$PROMPT_LEN_SP
osl=$NEW_TOKENS_SP

echo "=hyper params end="

# dataset preparation
create_dataset() {
    echo "RUN: create dataset"
    for i in $isl; do
        for o in $osl; do
            echo "dataset preparation, isl=$i, osl=$o"
            dataset_file="${WROOT}/dataset/${MODEL_NAME}/${MODEL_NAME}_${i}_${o}"
            dataset_dir=$(dirname "$dataset_file")
            mk_dir $dataset_dir 
            python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py --output=$dataset_file  --tokenizer=$hf_model_path token-norm-dist \
                --num-requests $requests \
                --input-mean=$i \
                --output-mean=$o \
                --input-stdev=0 \
                --output-stdev=0
        done
    done
}

create_dataset

echo "DONE: dataset created in: $dataset_dir"

# Call
#bash +x trtllm_createDataset.sh --model_dir $MODEL_DIR --requests 100  --prompt_len 100,1000,10000,50000,10000 --new_tokens 1000
