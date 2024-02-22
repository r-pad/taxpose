#!/bin/bash

echo "--------------------------------------------------------------------------------"
echo 'Evaluating pregrasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_put_knife_on_chopping_board"

python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/pregrasp data_root=/data wandb.group=rlbench_put_knife_on_chopping_board

echo "--------------------------------------------------------------------------------"
echo 'Evaluating grasp'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_put_knife_on_chopping_board"

python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/grasp data_root=/data wandb.group=rlbench_put_knife_on_chopping_board

echo "--------------------------------------------------------------------------------"
echo 'Evaluating lift'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_put_knife_on_chopping_board"

python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/lift data_root=/data wandb.group=rlbench_put_knife_on_chopping_board

echo "--------------------------------------------------------------------------------"
echo 'Evaluating place'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_put_knife_on_chopping_board"

python scripts/eval_metrics.py --config-name commands/rlbench/put_knife_on_chopping_board/taxpose_all/precision_eval/place data_root=/data wandb.group=rlbench_put_knife_on_chopping_board
