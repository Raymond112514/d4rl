#!/usr/bin/env bash
# Launch 4 BC training runs in parallel: unconditional/goal_conditioned x 50%/25% data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
mkdir -p logs

launch_run() {
  local mode="$1"
  local fraction="$2"
  local name="${mode}_ac1_${fraction}"
  echo "Launching ${name}..."
  nohup "${PYTHON}" train.py \
    --mode "${mode}" \
    --ac-chunk 1 \
    --data-fraction "${fraction}" \
    --wandb-run-name "${name}" \
    > "logs/${name}.log" 2>&1 &
  echo "  pid=$!  log=logs/${name}.log"
}

launch_run unconditional 0.5
launch_run goal_conditioned 0.5
launch_run unconditional 0.25
launch_run goal_conditioned 0.25

echo "All 4 runs launched. Checkpoints:"
echo "  checkpoints/unconditional_ac1_0.5"
echo "  checkpoints/goal_conditioned_ac1_0.5"
echo "  checkpoints/unconditional_ac1_0.25"
echo "  checkpoints/goal_conditioned_ac1_0.25"
