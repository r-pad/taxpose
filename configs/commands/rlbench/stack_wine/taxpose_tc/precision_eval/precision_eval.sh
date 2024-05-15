#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating pregrasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/pregrasp data_root=/data wandb.group=rlbench_stack_wine $@"

python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/pregrasp data_root=/data wandb.group=rlbench_stack_wine $@
echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/grasp data_root=/data wandb.group=rlbench_stack_wine $@"

python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/grasp data_root=/data wandb.group=rlbench_stack_wine $@
echo "--------------------------------------------------------------------------------"
echo 'Evaluating lift'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/lift data_root=/data wandb.group=rlbench_stack_wine $@"

python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/lift data_root=/data wandb.group=rlbench_stack_wine $@
echo "--------------------------------------------------------------------------------"
echo 'Evaluating preplace'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/preplace data_root=/data wandb.group=rlbench_stack_wine $@"

python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/preplace data_root=/data wandb.group=rlbench_stack_wine $@
echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/place data_root=/data wandb.group=rlbench_stack_wine $@"

python scripts/eval_metrics.py --config-name commands/rlbench/stack_wine/taxpose_tc/precision_eval/place data_root=/data wandb.group=rlbench_stack_wine $@
