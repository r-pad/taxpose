#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating pregrasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_take_money_out_safe $@"

python scripts/eval_metrics.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_take_money_out_safe $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_take_money_out_safe $@"

python scripts/eval_metrics.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_take_money_out_safe $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating lift'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_take_money_out_safe $@"

python scripts/eval_metrics.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_take_money_out_safe $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_take_money_out_safe $@"

python scripts/eval_metrics.py --config-name commands/rlbench/take_money_out_safe/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_take_money_out_safe $@
