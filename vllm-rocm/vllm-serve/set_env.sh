#!/bin/sh

# https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-os-settings
echo 0 > /proc/sys/kernel/numa_balancing
cat /proc/sys/kernel/numa_balancing
export HIP_FORCE_DEV_KERNARG=1
# To use CK FA
export VLLM_USE_TRITON_FLASH_ATTN=0
#export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
