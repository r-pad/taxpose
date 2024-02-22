#!/bin/bash

echo "------------------------------------------------------------------------------------------------------------------------"
echo "Running precision evaluation for take_umbrella_out_of_umbrella_stand task with taxpose_all model, pregasp"
echo "------------------------------------------------------------------------------------------------------------------------"
python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/pregasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

echo "------------------------------------------------------------------------------------------------------------------------"
echo "Running precision evaluation for take_umbrella_out_of_umbrella_stand task with taxpose_all model, grasp"
echo "------------------------------------------------------------------------------------------------------------------------"
python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

echo "------------------------------------------------------------------------------------------------------------------------"
echo "Running precision evaluation for take_umbrella_out_of_umbrella_stand task with taxpose_all model, lift"
echo "------------------------------------------------------------------------------------------------------------------------"
python scripts/eval_metrics.py --config-name commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand
