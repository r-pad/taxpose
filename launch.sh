#!/bin/bash

# This is a script that should take in three arguments:
# 1. the platfrom to run on (e.g. "autobot" or "local")
# 2. the index of which GPU to use
# 3. the command and arguments to run

# Example usage:
# ./launch_autobot.sh autobot 0 python scripts/train_residual_flow.py

# Get the first argument:
PLATFORM=$1

# Get the second argument:
GPU_INDEX=$2
shift
shift

# Get the third argument:
COMMAND=$@


echo Platform: $PLATFORM
echo GPU Index: $GPU_INDEX
echo Command: $COMMAND


# If the platform is "autobot", then we need to use singularity to run the command.
if [ $PLATFORM == "autobot" ]; then
    echo "Running on autobot"

    # Run on signularity.
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU_INDEX \
    SINGULARITYENV_WANDB_DOCKER_IMAGE=taxpose \
    singularity exec \
    --nv \
    -B /home/$(whoami)/code/rpad/taxpose:/opt/$(whoami)/code \
    -B /scratch/$(whoami)/data:/data \
    -B /scratch/$(whoami)/logs:/opt/logs \
    -B /scratch/$(whoami)/artifacts:/opt/artifacts \
    docker://beisner/taxpose \
    $COMMAND \
    log_dir=/opt/logs \
    data_dir=/data \
    wandb.artifact_dir=/opt/artifacts \


# If the platform is "local", then we can just run the command.
elif [ $PLATFORM == "local" ]; then
    echo "Running locally"

    CUDA_VISIBLE_DEVICES=$GPU_INDEX \
    WANDB_DOCKER_IMAGE=taxpose \
    $COMMAND

else
    echo "Platform not recognized"
fi