#!/bin/sh

#wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
#!/bin/sh
#BM_S=/app/vllm/benchmarks/benchmark_serving.py
MODEL=Llama-3.3-70B-Instruct
DATASET=./ShareGPT_V3_unfiltered_cleaned_split.json

#python3 $BM_S --backend vllm \
vllm bench serve \
  --backend openai \
  --model $MODEL \
  --tokenizer /data/$MODEL \
  --dataset-name sharegpt \
  --dataset-path $DATASET \
  --num-prompts 200 \
  --host 10.10.38.14 \
  --port 31436 \
  --endpoint /v1/completions \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --save-result \
  2>&1 | tee benchmark_serving.log


echo "### Serving Benchmarks" >> benchmark_results.md
sed -n '1p' benchmark_serving.log >> benchmark_results.md # first line
sed -n '2p' benchmark_serving.log >> benchmark_results.md
sed -n '5p' benchmark_serving.log >> benchmark_results.md
sed -n '6p' benchmark_serving.log >> benchmark_results.md
echo "" >> benchmark_results.md
echo '```' >> benchmark_results.md
tail -n 25 benchmark_serving.log >> benchmark_results.md
echo '```' >> benchmark_results.md

