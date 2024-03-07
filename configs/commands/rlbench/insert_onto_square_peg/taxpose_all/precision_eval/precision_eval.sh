#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/insert_onto_square_peg/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_insert_onto_square_peg $@"

python scripts/eval_metrics.py --config-name commands/rlbench/insert_onto_square_peg/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_insert_onto_square_peg $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/insert_onto_square_peg/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_insert_onto_square_peg $@"

python scripts/eval_metrics.py --config-name commands/rlbench/insert_onto_square_peg/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_insert_onto_square_peg $@
