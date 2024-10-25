#!/bin/bash

GEN_DATASET=0
RUN_CONVERT=1
RUN_BUILD=1
RUN_BM=1

MODEL_ROOT="/ws/m0/"
#MODEL_NAME="Mixtral-8x7B-Instruct-v0.1"
MODEL_NAME="Meta-Llama-3.1-405B-Instruct"
MODEL_PATH=${MODEL_ROOT}/${MODEL_NAME}
DS_ROOT="/ws/m1/dataset"
CKPT_ROOT="/ws/m1/trtllm_ckpt"
ENG_ROOT="/ws/m1/trtllm_engines"

echo -e "\nRUN: trtllm-gptManager benchmark, Model: ${MODEL_NAME}\n"

if [ $GEN_DATASET -eq 1 ]; then
    echo "=> gen dataset"
    ./trtllm_createDataset.sh --model_dir "${MODEL_PATH}" \
    --requests 100  \
    --prompt_len 100,1000,10000,50000,100000 \
    --new_tokens 1000
fi

if [ $RUN_CONVERT -eq 1 ]; then
    echo "=> convert"
    python3 /app/tensorrt_llm/examples/llama/convert_checkpoint.py \
    --model_dir "${MODEL_PATH}"  \
    --output_dir "${CKPT_ROOT}/${MODEL_NAME}" \
    --dtype bfloat16 \
    --use_fp8_rowwise \
    --tp_size 8 \
    --pp_size 1 \
    --load_by_shard \
    --workers 8
fi

# '--max_num_tokens 8192' work fine with BM when isl<=8192
if [ $RUN_BUILD -eq 1 ]; then
    echo "=> build"
    trtllm-build --checkpoint_dir "${CKPT_ROOT}/${MODEL_NAME}" \
    --output_dir "${ENG_ROOT}/${MODEL_NAME}" \
    --max_num_tokens 11000 \
    --max_input_len 100000 \
    --max_seq_len 101000 \
    --use_paged_context_fmha enable \
    --workers 8
fi

if [ $RUN_BM -eq 1 ]; then
    echo "=> test"
    cd /ws/m1/gptManagerBenchmark
    ./autoBM_test.sh --engine "${ENG_ROOT}/${MODEL_NAME}" --dataset "${DS_ROOT}/${MODEL_NAME}"
else
    echo "Auto BM is disabled."
fi
