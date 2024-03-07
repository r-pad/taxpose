#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_toilet_roll_on_stand/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_put_toilet_roll_on_stand $@"

python scripts/eval_metrics.py --config-name commands/rlbench/put_toilet_roll_on_stand/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_put_toilet_roll_on_stand $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_toilet_roll_on_stand/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_put_toilet_roll_on_stand $@"

python scripts/eval_metrics.py --config-name commands/rlbench/put_toilet_roll_on_stand/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_put_toilet_roll_on_stand $@
