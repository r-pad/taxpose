# This is a script that should take in two arguments:
# 1. the index of which GPU to use
# 2. the command and arguments to run

# Example usage:
# ./launch_autobot.sh 0 python scripts/train_residual_flow.py log_dir=/opt/logs

# Get the first argument:
GPU_INDEX=$1
shift

# Get the second argument:
COMMAND=$@


SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU_INDEX \
SINGULARITYENV_WANDB_DOCKER_IMAGE=taxpose \
singularity exec \
--nv \
-B /scratch/$(whoami)/data:/data \
-B /scratch/$(whoami)/logs:/opt/logs \
docker://beisner/taxpose \
$COMMAND
