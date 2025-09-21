#!/bin/bash

# --- 脚本配置 ---
PROGRAM_TO_RUN="python main.py"
GPU_DEVICE_ID="0"
NUM_RUNS=3

# --- 脚本主体 ---
echo "--- 脚本开始运行 ---"

for ((i=1; i<=NUM_RUNS; i++))
do
    echo "--- 第 $i 次运行 ---"

    # 使用 CUDA_VISIBLE_DEVICES 环境变量指定显卡，并将输出通过管道传输给 tail 命令
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID $PROGRAM_TO_RUN | tail -n 10
    echo ""
done

echo "--- 所有运行已完成 ---"