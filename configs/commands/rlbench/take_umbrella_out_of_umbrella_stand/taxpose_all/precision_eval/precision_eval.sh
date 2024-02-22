#!/bin/bash

echo "--------------------------------------------------------------------------------"
echo 'Evaluating pregasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/pregasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand"

python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/pregasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand"

python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

echo "--------------------------------------------------------------------------------"
echo 'Evaluating lift'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand"

python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand
