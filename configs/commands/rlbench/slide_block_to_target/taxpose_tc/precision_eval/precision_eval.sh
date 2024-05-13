#!/bin/bash

set -e

echo 'Running precision eval'

echo "--------------------------------------------------------------------------------"
echo 'Evaluating preslide'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/precision_eval/preslide data_root=/data wandb.group=rlbench_slide_block_to_target $@"

python scripts/eval_metrics.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/precision_eval/preslide data_root=/data wandb.group=rlbench_slide_block_to_target $@

echo "--------------------------------------------------------------------------------"
echo 'Evaluating postslide'
echo "--------------------------------------------------------------------------------"
echo "python scripts/eval_metrics.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/precision_eval/postslide data_root=/data wandb.group=rlbench_slide_block_to_target $@"

python scripts/eval_metrics.py --config-name commands/rlbench/slide_block_to_target/taxpose_tc/precision_eval/postslide data_root=/data wandb.group=rlbench_slide_block_to_target $@
