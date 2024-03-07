#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/solve_puzzle/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_solve_puzzle $@"

python scripts/eval_metrics.py --config-name commands/rlbench/solve_puzzle/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_solve_puzzle $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/solve_puzzle/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_solve_puzzle $@"

python scripts/eval_metrics.py --config-name commands/rlbench/solve_puzzle/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_solve_puzzle $@
