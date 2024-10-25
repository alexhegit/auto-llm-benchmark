#!/bin/bash

GEN_DATASET=1
RUN_CONVERT=1
RUN_BUILD=1
RUN_BM=1

MODEL_ROOT="/ws/m0/"
#MODEL_NAME="Mixtral-8x22B-Instruct-v0.1"
MODEL_NAME="Mixtral-8x22B-Instruct-v0.1-FP8"
MODEL_PATH=${MODEL_ROOT}/${MODEL_NAME}
DS_ROOT="/ws/m1/dataset"
CKPT_ROOT="/ws/m1/trtllm_ckpt"
ENG_ROOT="/ws/m1/trtllm_engines"

echo -e "\nRUN: trtllm-gptManager benchmark, Model: ${MODEL_NAME}, TP8\n"

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
    --tp_size 8 \
    --moe_tp_size 2 \
    --moe_ep_size 4
fi

# '--max_num_tokens 60000' work fine with BM when (isl+osl)<=60000
# --gemm_plugin bfloat16
if [ $RUN_BUILD -eq 1 ]; then
    echo "=> build"
    trtllm-build --checkpoint_dir "${CKPT_ROOT}/${MODEL_NAME}" \
    --output_dir "${ENG_ROOT}/${MODEL_NAME}" \
    --max_num_tokens 101000 \
    --max_input_len 100000 \
    --max_seq_len 101000
    --workers 8
fi

if [ $RUN_BM -eq 1 ]; then
    echo "=> test"
    ./autoBM_gptM.sh --engine "${ENG_ROOT}/${MODEL_NAME}" --dataset "${DS_ROOT}/${MODEL_NAME}"
else
    echo "Auto BM is disabled."
fi

