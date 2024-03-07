#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/place_hanger_on_rack/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_place_hanger_on_rack $@"

python scripts/eval_metrics.py --config-name commands/rlbench/place_hanger_on_rack/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_place_hanger_on_rack $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/place_hanger_on_rack/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_place_hanger_on_rack $@"

python scripts/eval_metrics.py --config-name commands/rlbench/place_hanger_on_rack/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_place_hanger_on_rack $@
