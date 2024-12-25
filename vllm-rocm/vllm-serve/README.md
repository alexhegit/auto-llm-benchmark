# Guide to benchmark the vllm serving

## Start the container

- Pull the docker image
```bash
docker pull rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
```

- Start the rocm/vllm container
**reference cmd**
```bash
alias drun="docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 256g --net host -v $PWD:/ws -v /data:/data --entrypoint /bin/bash --env HUGGINGFACE_HUB_CACHE=/data/llm -w /ws"

drun --name vllm-bm 
```

## Benchmark

Run the steps below in the container

- Set env for better performance

`set_env.sh`
```bash
#!/bin/sh

# refer to https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-os-settings
echo 0 > /proc/sys/kernel/numa_balancing
cat /proc/sys/kernel/numa_balancing
export HIP_FORCE_DEV_KERNARG=1
# To use CK FA
export VLLM_USE_TRITON_FLASH_ATTN=0
#export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
```

- Start the `vllm serve`

`run_vllm_serve.sh`
```bash
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
```

- Verify the `vllm serve`

```
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/completions \
	-H "Content-Type: application/json" \
	-d '{
        "model": "/data/llm/Meta-Llama-3.1-8B/",
	"prompt": "San Francisco is a",
	"max_tokens": 7,
	"temperature": 0
}'
```

- Run benchmark

Download the dataset
```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

`benchmark_serving.sh`
  ```bash
#!/bin/sh
BM_S=/app/vllm/benchmarks/benchmark_serving.py
MODEL=/data/llm/Meta-Llama-3.1-8B/
DATASET=./ShareGPT_V3_unfiltered_cleaned_split.json

python3 $BM_S --backend vllm \
	--model $MODEL \
	--tokenizer $MODEL \
	--dataset-name sharegpt \
	--dataset-path $DATASET \
	--num-prompts 200 \
	--port 8000 \
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
  ```

The results will be saved into benchmark_results.md as this,

```
### Serving Benchmarks
WARNING 12-25 08:35:21 rocm.py:17] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=None, model='/data/llm/Meta-Llama-3.1-8B/', tokenizer='/data/llm/Meta-Llama-3.1-8B/', best_of=1, use_beam_search=False, num_prompts=200, logprobs=None, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=True, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl,e2el', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None)
Traffic request rate: inf
Maximum request concurrency: None

============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  8.63      
Total input tokens:                      42659     
Total generated tokens:                  26776     
Request throughput (req/s):              23.18     
Output token throughput (tok/s):         3103.24   
Total Token throughput (tok/s):          8047.27   
---------------Time to First Token----------------
Mean TTFT (ms):                          1441.22   
Median TTFT (ms):                        2112.70   
P99 TTFT (ms):                           2134.04   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.74      
Median TPOT (ms):                        9.70      
P99 TPOT (ms):                           15.87     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.49      
Median ITL (ms):                         9.38      
P99 ITL (ms):                            16.01     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2297.19   
Median E2EL (ms):                        3386.84   
P99 E2EL (ms):                           8546.65   
==================================================
```

## Reference

- https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/performance-benchmarks-descriptions.md
- https://simon-mo-workspace.observablehq.cloud/vllm-dashboard-v0/perf
- https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-os-settings
