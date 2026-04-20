#!/bin/bash
# 实时监控GPU显存和系统内存使用情况

echo "开始监控GPU和内存使用情况..."
echo "按Ctrl+C退出"
echo "========================================"

while true; do
    clear
    date
    echo "========================================"
    echo "GPU 显存使用情况："
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU%s: %s - %s/%s MB (GPU利用率: %s%%)\n", $1, $2, $3, $4, $5}'

    echo ""
    echo "========================================"
    echo "系统内存使用情况："
    free -h | grep -E "Mem:|Swap:"

    echo ""
    echo "========================================"
    echo "评估进程数："
    ps aux | grep "src/evaluation.py" | grep -v grep | wc -l

    sleep 2
done
