#!/bin/bash
set -euo pipefail

MEMS=("512m" "1g" "2g")
CPUS=("0.5" "1" "2")
MODELS=("rf" "xgb" "cnn")

mkdir -p results

for mem in "${MEMS[@]}"; do
  for cpu in "${CPUS[@]}"; do
    echo "=== Config: MEM=$mem CPU=$cpu ==="
    for model in "${MODELS[@]}"; do
      run_id="${model}_${mem}_${cpu}"
      outfile="results/${run_id}.jsonl"

      echo "Running $run_id ..."
      docker run --rm \
        --memory="$mem" --cpus="$cpu" \
        -e MEM_LIMIT="$mem" -e CPU_LIMIT="$cpu" -e RUN_ID="$run_id" \
        edge-benchmark:latest \
        python scripts/benchmark.py "$model" | tee "$outfile"
    done
  done
done

echo "All done. Results in ./results/*.jsonl"
