#!/bin/bash

echo "--------------------------------------------------------------------------------"
echo 'Evaluating prepush'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/push_button/taxpose_all/precision_eval/prepush data_root=/data wandb.group=rlbench_push_button"

python scripts/eval_metrics.py --config-name commands/rlbench/push_button/taxpose_all/precision_eval/prepush data_root=/data wandb.group=rlbench_push_button

echo "--------------------------------------------------------------------------------"
echo 'Evaluating postpush'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/push_button/taxpose_all/precision_eval/postpush data_root=/data wandb.group=rlbench_push_button"

python scripts/eval_metrics.py --config-name commands/rlbench/push_button/taxpose_all/precision_eval/postpush data_root=/data wandb.group=rlbench_push_button
