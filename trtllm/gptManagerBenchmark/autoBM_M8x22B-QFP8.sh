#!/bin/bash

GEN_DATASET=0
RUN_CONVERT=1
RUN_BUILD=1
RUN_BM=1

MODEL_ROOT="/ws/m0"
MODEL_NAME="Mixtral-8x22B-Instruct-v0.1"
MODEL_PATH=${MODEL_ROOT}/${MODEL_NAME}
DS_ROOT="/ws/m1/dataset"
CKPT_ROOT="/ws/m1/trtllm_ckpt"
ENG_ROOT="/ws/m1/trtllm_engines"

echo -e "\nRUN: trtllm-gptManager benchmark, Model: ${MODEL_NAME}, QFP8, TP8\n"

if [ $GEN_DATASET -eq 1 ]; then
    echo "=> gen dataset"
    ./trtllm_createDataset.sh --model_dir "${MODEL_PATH}" \
    --requests 100  \
    --prompt_len 100,1000,10000,50000,100000 \
    --new_tokens 1000
fi

# Build with FP8 PTQ
if [ $RUN_CONVERT -eq 1 ]; then
    echo "=> convert with FP8 PTQ, TP8"
    python3 /app/tensorrt_llm/examples/quantization/quantize.py \
    --model_dir "${MODEL_PATH}"  \
    --output_dir "${CKPT_ROOT}/${MODEL_NAME}" \
    --dtype float16 \
    --qformat fp8 \
    --calib_size 512 \
    --tp_size 8
fi

# '--max_num_tokens 60000' work fine with BM when (isl+osl)<60000
if [ $RUN_BUILD -eq 1 ]; then
    echo "=> build"
    trtllm-build --checkpoint_dir "${CKPT_ROOT}/${MODEL_NAME}" \
    --output_dir "${ENG_ROOT}/${MODEL_NAME}" \
    --max_num_tokens 101000 \
    --max_input_len 100000 \
    --max_seq_len 101000 \
    --workers 8
fi

if [ $RUN_BM -eq 1 ]; then
    echo "=> test"
    ./autoBM_gptM.sh --engine "${ENG_ROOT}/${MODEL_NAME}" --dataset "${DS_ROOT}/${MODEL_NAME}"
else
    echo "Auto BM is disabled."
fi
