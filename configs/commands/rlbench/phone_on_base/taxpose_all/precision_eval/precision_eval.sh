#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating pregrasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/phone_on_base/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_phone_on_base $@"

python scripts/eval_metrics.py --config-name commands/rlbench/phone_on_base/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_phone_on_base $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/phone_on_base/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_phone_on_base $@"

python scripts/eval_metrics.py --config-name commands/rlbench/phone_on_base/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_phone_on_base $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating lift'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/phone_on_base/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_phone_on_base $@"

python scripts/eval_metrics.py --config-name commands/rlbench/phone_on_base/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_phone_on_base $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/phone_on_base/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_phone_on_base $@"

python scripts/eval_metrics.py --config-name commands/rlbench/phone_on_base/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_phone_on_base $@
