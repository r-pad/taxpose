#!/bin/bash

echo "--------------------------------------------------------------------------------"
echo 'Evaluating pregrasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_put_money_in_safe"

python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_put_money_in_safe

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_put_money_in_safe"

python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_put_money_in_safe

echo "--------------------------------------------------------------------------------"
echo 'Evaluating lift'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_put_money_in_safe"

python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_put_money_in_safe

echo "--------------------------------------------------------------------------------"
echo 'Evaluating preplace'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/preplace data_root=/data wandb.group=rlbench_put_money_in_safe"

python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/preplace data_root=/data wandb.group=rlbench_put_money_in_safe

echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_put_money_in_safe"

python scripts/eval_metrics.py --config-name commands/rlbench/put_money_in_safe/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_put_money_in_safe
