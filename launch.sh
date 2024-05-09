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

    # For the following directories, check to see if they exist. If they don't, create them. Use an array.
    # Directories to check:
    DIRECTORIES=("/scratch/$(whoami)/data" "/scratch/$(whoami)/logs" "/scratch/$(whoami)/artifacts" "/scratch/$(whoami)/.config" "/scratch/$(whoami)/tmp" "/scratch/$(whoami)/home")

    for DIRECTORY in "${DIRECTORIES[@]}"; do
        if [ ! -d $DIRECTORY ]; then
            mkdir -p $DIRECTORY
        fi
    done

    # Run on signularity.
    APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPU_INDEX \
    APPTAINERENV_WANDB_DOCKER_IMAGE=taxpose \
    APPTAINERENV_MPLCONFIGDIR=/opt/.config \
    apptainer run \
    --nv \
    --no-mount hostfs \
    --pwd /opt/$(whoami)/code \
    --workdir /opt/tmp \
    -B /home/$(whoami)/code/rpad/taxpose:/opt/$(whoami)/code \
    -B /scratch/$(whoami)/data:/data \
    -B /scratch/$(whoami)/logs:/opt/logs \
    -B /scratch/$(whoami)/artifacts:/opt/artifacts \
    -B /scratch/$(whoami)/.config:/opt/.config \
    -B /scratch/$(whoami)/tmp:/tmp \
    -B /scratch/$(whoami)/home:/home/$(whoami) \
    docker://beisner/taxpose \
    $COMMAND \
    log_dir=/opt/logs \
    data_root=/data \
    wandb.artifact_dir=/opt/artifacts \

# If the platform is "local-docker", then we need to use docker to run the command.
elif [ $PLATFORM == "local-docker" ]; then
    echo "Running locally with docker"

    docker run \
    --gpus "device=$GPU_INDEX" \
    -it \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e WANDB_DOCKER_IMAGE=taxpose \
    -v /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    -v /data/datasets/:/data \
    -v /home/eycai/Documents/taxpose/artifacts:/opt/artifacts \
    -v /home/eycai/Documents/taxpose/logs:/opt/logs \
    -v /home/eycai/Documents/taxpose:/opt/baeisner/code \
    beisner/taxpose \
    $COMMAND \
    log_dir=/opt/logs \
    data_root=/data \
    wandb.artifact_dir=/opt/artifacts

elif [ $PLATFORM == "local-apptainer" ]; then
    echo "Running locally with apptainer"

    APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPU_INDEX \
    APPTAINERENV_WANDB_DOCKER_IMAGE=taxpose \
    APPTAINERENV_MPLCONFIGDIR=/opt/.config \
    APPTAINERENV_VGL_DEVICE=egl$GPU_INDEX \
    APPTAINERENV_PYENV_VERSION= \
    apptainer run \
    --nv \
    --no-mount hostfs \
    --pwd /opt/$(whoami)/code \
    --contain \
    -B /home/$(whoami)/code/rpad/taxpose:/opt/$(whoami)/code \
    -B /home/$(whoami)/datasets:/data \
    -B /home/$(whoami)/code/rpad/taxpose/logs:/opt/logs \
    -B /home/$(whoami)/code/rpad/taxpose/artifacts:/opt/artifacts \
    -B /home/$(whoami)/.config:/opt/.config \
    -B /home/$(whoami)/.tmp:/tmp \
    -B /home/$(whoami)/tmp_home:/home/$(whoami) \
    -B /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    docker://beisner/taxpose \
    $COMMAND \
    log_dir=/opt/logs \
    data_root=/data \
    wandb.artifact_dir=/opt/artifacts

# If the platform is "local", then we can just run the command.
elif [ $PLATFORM == "local" ]; then
    echo "Running locally"

    CUDA_VISIBLE_DEVICES=$GPU_INDEX \
    WANDB_DOCKER_IMAGE=taxpose \
    $COMMAND

else
    echo "Platform not recognized"
fi
