
Downloading the data.

```bash
singularity exec \
--nv \
--pwd /opt/$(whoami)/code \
-B /scratch/$(whoami)/data:/opt/data \
docker://beisner/taxpose \
/opt/code/scripts/download_data.sh \
    /opt/data/ndf
```


Pretraining.

```bash
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
SINGULARITYENV_WANDB_DOCKER_IMAGE=taxpose \
singularity exec \
--nv \
-B /scratch/$(whoami)/data:/data \
-B /scratch/$(whoami)/logs:/opt/logs \
docker://beisner/taxpose \
python scripts/train_residual_flow.py \
    log_dir=/opt/logs
```



Training the model.

```bash
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
SINGULARITYENV_WANDB_DOCKER_IMAGE=taxpose \
singularity exec \
--nv \
--pwd /opt/$(whoami)/code \
-B /scratch/$(whoami)/data:/opt/data \
-B /scratch/$(whoami)/logs:/opt/logs \
docker://beisner/taxpose \
python scripts/train_residual_flow.py \
    task=mug_grasp \
    model=taxpose \
    +mode=train \
    benchmark.dataset_root=/opt/data \
    log_dir=/opt/logs
```
