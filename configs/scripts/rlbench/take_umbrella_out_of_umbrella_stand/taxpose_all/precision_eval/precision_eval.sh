#!/bin/bash

--------------------------------------------------------------------------------
echo 'Evaluating pregasp'
--------------------------------------------------------------------------------
echo python scripts/eval_metrics.py commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/pregasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

python scripts/eval_metrics.py commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/pregasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

--------------------------------------------------------------------------------
echo 'Evaluating grasp'
--------------------------------------------------------------------------------
echo python scripts/eval_metrics.py commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

python scripts/eval_metrics.py commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

--------------------------------------------------------------------------------
echo 'Evaluating lift'
--------------------------------------------------------------------------------
echo python scripts/eval_metrics.py commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand

python scripts/eval_metrics.py commands/rlbench/take_umbrella_out_of_umbrella_stand/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_take_umbrella_out_of_umbrella_stand
