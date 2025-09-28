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

DIMG=rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
drun --name vllm-bm $DIMG
```

## Benchmark Serving

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

- Start the sever `vllm serve`

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

### Benchmark with dataset `ShareGPT`

- Run benchmark client

Download the dataset
```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

`benchmark_serving.sh`
  
```bash
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
```

The results will be saved into benchmark_results.md as this,

```
### Serving Benchmarks
INFO 09-28 06:18:56 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7fe925f29120>, seed=0, num_prompts=200, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', no_oversample=False, custom_output_len=256, custom_skip_chat_template=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, blazedit_min_distance=0.0, blazedit_max_distance=1.0, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, label=None, backend='openai', endpoint_type=None, base_url=None, host='10.10.38.14', port=31436, endpoint='/v1/completions', header=None, max_concurrency=None, model='Llama-3.3-70B-Instruct', tokenizer='/data/Llama-3.3-70B-Instruct', use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=True, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl,e2el', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |          | 00:01 elapsed, 22:05:52 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|__________| 200/200 [00:25<00:00,  7.70it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  25.98
Total input tokens:                      42659
Total generated tokens:                  43037
Request throughput (req/s):              7.70
Output token throughput (tok/s):         1656.66
Peak output token throughput (tok/s):    3494.00
Peak concurrent requests:                200.00
Total Token throughput (tok/s):          3298.77
---------------Time to First Token----------------
Mean TTFT (ms):                          4062.30
Median TTFT (ms):                        4279.80
P99 TTFT (ms):                           7126.81
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.76
Median TPOT (ms):                        37.46
P99 TPOT (ms):                           74.30
---------------Inter-token Latency----------------
Mean ITL (ms):                           33.90
Median ITL (ms):                         32.67
P99 ITL (ms):                            52.24
----------------End-to-end Latency----------------
Mean E2EL (ms):                          11357.96
Median E2EL (ms):                        9917.56
P99 E2EL (ms):                           25830.99
==================================================

```

### Benchmark with random

- Run benchmark client
Here is the example with input-len=1000, output-len=1000 and request(--num-prompts)=1000, --request-rate=250.

`benchmark_serving_random.sh`
```bash
#!/bin/sh
BM_S=/app/vllm/benchmarks/benchmark_serving.py
MODEL=/data/llm/Meta-Llama-3.1-8B/

python3 $BM_S --backend vllm \
        --model $MODEL \
        --tokenizer $MODEL \
        --dataset-name random \
        --random-input-len 1000 \
        --random-output-len 1000 \
        --num-prompts 1000 \
	--request-rate 250 \
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


The result looks like,

```
### Serving Benchmarks
WARNING 12-26 09:48:53 rocm.py:17] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Namespace(backend='vllm', base_url=None, host='localhost', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='random', dataset_path=None, max_concurrency=None, model='/data/llm/Meta-Llama-3.1-8B/', tokenizer='/data/llm/Meta-Llama-3.1-8B/', best_of=1, use_beam_search=False, num_prompts=1000, logprobs=None, request_rate=250.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=True, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl,e2el', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1000, random_output_len=1000, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None)
Traffic request rate: 250.0
Maximum request concurrency: None

============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  215.83
Total input tokens:                      1000000
Total generated tokens:                  882429
Request throughput (req/s):              4.63
Output token throughput (tok/s):         4088.47
Total Token throughput (tok/s):          8721.66
---------------Time to First Token----------------
Mean TTFT (ms):                          31811.46
Median TTFT (ms):                        34969.71
P99 TTFT (ms):                           57809.12
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1852.87
Median TPOT (ms):                        152.41
P99 TPOT (ms):                           67678.21
---------------Inter-token Latency----------------
Mean ITL (ms):                           168.57
Median ITL (ms):                         66.04
P99 ITL (ms):                            1008.59
----------------End-to-end Latency----------------
Mean E2EL (ms):                          165852.29
Median E2EL (ms):                        178415.97
P99 E2EL (ms):                           210843.04
==================================================
```

### fw-ai/benchmark
Fireworks-AI has a tool <https://github.com/fw-ai/benchmark> which is a wraper of [locust](https://locust.io/) for easy usage. We could use it as the http client to test the vLLM serve.

- Start vLLM Serve 
```
MODEL=/data/llm/meta-llama/Llama-3.1-8B-Instruct/
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
- Client
Installation
```
git clone https://github.com/fw-ai/benchmark
cd ./benchmark/llm_bench
pip install -r requirements.txt
```
Benchmark
```
MODEL=/data/llm/meta-llama/Llama-3.1-8B-Instruct/

locust --provider vllm -H http://localhost:8000 --headless \
	--summary-file vllm_locust.csv --tokenizer $MODEL  \
	--qps 1.0 -u 100 -r 100 -p 1000 -o 100 -t 2m
```
Result
```
Type     Name                                                                          # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /v1/completions                                                                  119     0(0.00%) |      8       6       9      8 |    1.01        0.00
GET      /v1/models                                                                         0     0(0.00%) |      0       0       0      0 |    0.00        0.00
METRIC   latency_per_char                                                                 118     0(0.00%) |      1       0       1      1 |    1.00        0.00
METRIC   latency_per_token                                                                  0     0(0.00%) |      0       0       0      0 |    0.00        0.00
METRIC   num_tokens                                                                         0     0(0.00%) |      0       0       0      0 |    0.00        0.00
METRIC   prompt_tokens                                                                      0     0(0.00%) |      0       0       0      0 |    0.00        0.00
METRIC   time_to_first_token                                                              118     0(0.00%) |     57      55      59     57 |    1.00        0.00
METRIC   total_latency                                                                    118     0(0.00%) |    572     567     574    570 |    1.00        0.00
--------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated                                                                       473     0(0.00%) |    159       0     574     10 |    4.01        0.00

Response time percentiles (approximated)
Type     Name                                                                                  50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
POST     /v1/completions                                                                         8      8      8      8      9      9      9      9     10     10     10    119
METRIC   latency_per_char                                                                        1      1      1      1      1      2      2      2      2      2      2    118
METRIC   time_to_first_token                                                                    57     58     58     58     58     58     58     58     59     59     59    118
METRIC   total_latency                                                                         570    570    570    570    570    570    570    570    570    570    570    118
--------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated                                                                             10     58     59    570    570    570    570    570    570    570    570    473

=================================== Summary ====================================
Provider                 : vllm
Model                    : /data/llm/meta-llama/Llama-3.1-8B-Instruct/
Prompt Tokens            : 0.0
Generation Tokens        : 100
Stream                   : True
Temperature              : 1.0
Logprobs                 : None
Concurrency              : QPS 1.0 constant
Time To First Token      : 57.354352397496925
Latency Per Token        : 0.0
Num Tokens               : 0.0
Total Latency            : 572.2289447814732
Num Requests             : 118
Qps                      : 0.9999774881355382
P50 Time To First Token  : 57
P90 Time To First Token  : 58
P99 Time To First Token  : 58
P99.9 Time To First Token: 59
P50 Total Latency        : 570.0
P90 Total Latency        : 570.0
P99 Total Latency        : 570.0
P99.9 Total Latency      : 570.0
================================================================================

```


**NOTE**

vLLM official provide the benchmark data at https://simon-mo-workspace.observablehq.cloud/vllm-dashboard-v0/perf

## Reference

- https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/performance-benchmarks-descriptions.md
- https://simon-mo-workspace.observablehq.cloud/vllm-dashboard-v0/perf
- https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-os-settings
- https://github.com/fw-ai/benchmark
- https://maddevs.io/writeups/first-performance-test-with-locust/
