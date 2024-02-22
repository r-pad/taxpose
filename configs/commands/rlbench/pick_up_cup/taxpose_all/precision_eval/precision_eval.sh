#!/bin/bash

echo "--------------------------------------------------------------------------------"
echo 'Evaluating pregrasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/pick_up_cup/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_pick_up_cup"

python scripts/eval_metrics.py --config-name commands/rlbench/pick_up_cup/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_pick_up_cup

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/pick_up_cup/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_pick_up_cup"

python scripts/eval_metrics.py --config-name commands/rlbench/pick_up_cup/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_pick_up_cup

echo "--------------------------------------------------------------------------------"
echo 'Evaluating lift'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/pick_up_cup/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_pick_up_cup"

python scripts/eval_metrics.py --config-name commands/rlbench/pick_up_cup/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_pick_up_cup
