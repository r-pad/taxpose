#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating reach'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/reach_target/taxpose_all/precision_eval/reach data_root=/data wandb.group=rlbench_reach_target $@"

python scripts/eval_metrics.py --config-name commands/rlbench/reach_target/taxpose_all/precision_eval/reach data_root=/data wandb.group=rlbench_reach_target $@
