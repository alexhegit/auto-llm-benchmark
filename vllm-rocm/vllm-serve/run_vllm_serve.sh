#!/bin/sh

#export CUDA_VISIBLE_DEVICES=7
#MODEL=/data/llm/meta-llama/Llama-3.1-8B-Instruct/
MODEL=/data/llm/Meta-Llama-3.1-8B/
TP=1

vllm serve $MODEL \
    --swap-space 16 \
    --disable-log-requests \
    --tensor-parallel-size $TP \
    --distributed-executor-backend mp \
    --num-scheduler-steps 10 \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill=False \
    --max-num-seqs 1000 \
    --max-model-len 8192 \
    --max-num-batched-tokens 65536
