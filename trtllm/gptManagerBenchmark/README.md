The auto benchmark is implemented base on https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/cpp/gptManagerBenchmark.cpp.

Please get details from https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/cpp/README.md

All test script verfified with trtllm v0.13 and 4 models both in BF16/FP16 and FP8.
1. Llama-3.1-405B
2. LLama-3.1-70B
3. Mixtral-8x22B
4. Mixtral-8x77B

Each model auto test has a dedicated script. 

Example: The `autoBM_L405B-FP8.sh` is for Llama-3.1-405B-FP8.
1. Run it ./autoBM_L405B-FP8.
It will run all the steps according the switch var defined in it.
```
GEN_DATASET=0
RUN_CONVERT=1
RUN_BUILD=1
RUN_BM=1
```
The full test process have 4 steps. 
- Create dataset
- Convert HF model to trtllm model checkpoint
- Build the checkpoint to engine of trtllm
- Run gptManagerBenchmark

2. You can auto test multiple modes like `autoBM-with-list.sh`

You may need to modify them with on your environment like model path, test cases(isl, osl, batchsize), results log and csv save path and different build engine configs, etc.
