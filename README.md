# TAX-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation

This is the official code release for our CoRL 2022 paper:

**TAX-Pose: Task-Specific Cross-Pose Estimation
for Robot Manipulation**
*Chuer Pan\*, Brian Okorn\*, Harry Zhang\*, Ben Eisner\*, David Held*
CoRL 2022

```
@inproceedings{pan2022taxpose,
    title       = {{TAX}-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation},
    author      = {Chuer Pan and Brian Okorn and Harry Zhang and Ben Eisner and David Held},
    booktitle   = {6th Annual Conference on Robot Learning},
    year        = {2022},
    url         = {https://openreview.net/forum?id=YmJi0bTfeNX}
}
```

Questions? Open an Issue, or send an email to:
ben [dot] a [dot] eisner [at] gmail [dot] com


# Reproduciblilty

Some of the eval scripts are not bit-for-bit reproducible even with random seeds. Some of the motion planning in the NDF repository depends on time elapsed, instead of number of steps. So predictions seem to be quite reproducible, but the exact motion planning trajectories are not (meaning that teleport scores are deterministic, but not the executed ones).

Some things we could try to fully make it reproducible:
* Make each trial fully independent (i.e. setup and teardown fully inside the loop).
* Fork the motion planning stack to incorporate a step-based time counter instead of a real-time counter.
* Recompile the IK with a random seed (this may not be necessary).
* Implement `move_ee_xyz` without using the realtime mode in `airobot`. This would involve doing the same kind of following that we do when setting joint positons.


# Installation.

This repository can be conveniently installed as a standard Python package, and used in downstream projects like any other dependency in any Python environment (i.e. virtualenv, conda, etc.). However, there are a few prerequisites for installation:

## Install dependencies.

Before installing `taxpose`, you'll need to make sure you have versions of the following dependencies installed:

* torch
* torch_geometric
* pytorch3d
* dgl

They have to be installed in a platform-specific manner, because they each contain different CUDA kernels that need to be complied...

IMPORTANT: The CUDA version must match between all these dependencies, eg. 11.3.

### `torch`

You can follow instructions [here](https://pytorch.org/get-started/locally/).

For our experiments, we installed torch 1.11 with cuda 11.3:

```
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

```

### `pytorch-geometric`

You can follow instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

For our experiments, we installed with the following command:

```
# pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 pyg_lib==0.1.0 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 pyg_lib==0.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### `pytorch3d`

Follow instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#3-install-wheels-for-linux).

We ran the following:

```
pip install fvcore iopath
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

```

### `dgl`

Follow instructions [here](https://www.dgl.ai/pages/start.html).

For our experiments, we ran:
```
pip install --pre dgl -f https://data.dgl.ai/wheels/cu113/repo.html
pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Install `taxpose`.

This part should be really simple:

### Option 1: Install from Github.

If you don't want to make any modifications, you can install directly.

```
pip install "taxpose @ git+https://github.com/r-pad/taxpose.git"
```

### Option 2: Install in editable mode.

If you want to make changes, you can install in editable mode.

```
git clone https://github.com/r-pad/taxpose.git
cd taxpose
pip install -e .
```

## Install `ndf_robot`.

For some reason, we need to install this in editable mode (can't just pip install it from github).

```
git submodule update --init --recursive
pip install -e third_party/ndf_robot
```

### Download the data.

```
cd third_party/ndf_robot
NDF_SOURCE_DIR=$PWD/src/ndf_robot ./scripts/download_obj_data.sh
NDF_SOURCE_DIR=$PWD/src/ndf_robot ./scripts/download_demo_demonstrations.sh
```

### Install the pybullet planning.

Following: https://github.com/anthonysimeonov/ndf_robot/tree/master#setup

This version of IKFast is evil and requires Python 2.7.

```
sudo apt-get install python-dev
cd third_party/ndf_robot/pybullet-planning/pybullet_tools/ikfast/franka_panda
python setup.py
```


# Code Structure

TODO(beisner): Clean this up.

TODO(beisner): Clean up the code structure a little bit, one day.

* `docs/`: (eventual) autogenerated documentation for the code
* `notebooks/`
    * `pm_placement.ipynb`: Simple visualization of the PM Placement dataset.
* `results/`: The pm evaluation script will dump CSVs of the results here.
* `scripts/`
    * `create_pm_dataset.py`: Script which will generate the cached version Partnet-Mobility Placement dataset.
    * `evaluate_ndf_mug.py`: Evaluate the NDF task on the mug.
    * `pretrain_embedding.py`: Pretrain embeddings for the NDF tasks.
    * `sample_action_surfaces.py`: Sample full point clouds for the action objects for PM Placement tasks.
    * `train_residual_flow.py`: Train TAX-Pose on the NDF tasks.
* `taxpose/`
    * `datasets/`: Dataset classes for the PM Placement and NDF tasks.
    * `models/`: Models. Downstream users will probably only be interested in `taxpose.models.taxpose.TAXPoseModel`.
    * `nets/`: Networks used by the models.
    * `training/`: Training code for the NDF tasks.
    * `utils/`: Utility functions.
    * `train_pm_placement.py`: Training code for the PM Placement tasks.

# I want to...

## ...use the PM Placement dataset.

TODO(beisner): Add instructions.

## ...use the TAX-Pose model code.

TODO(beisner): Add instructions.

## ...use a pre-trained TAX-Pose model.

Here are links to pre-trained models:

#### Download the pre-trained models for NDF mug.
```
bash download_trained_mug_models.sh
```

This will download the pre-trained models for NDF mug and save them at:
* `trained_models`
    * `ndf`
        * `arbitrary`
            * `place.ckpt`
            * `grasp.ckpt`
        * `upright`
            * `place.ckpt`
            * `grasp.ckpt`

TODO(beisner, chuer): Add links.

* NDF-Mug
* NDF-Bowl
* NDF-Bottle
* PM-Placement

# Reproduce our entire paper.

## Table 1: NDF Tasks

In our work, we compare against the performance of Neural Descriptor Fields on the task they released in their paper.

Original code can be found here: https://github.com/anthonysimeonov/ndf_robot

### Pretrain the embeddings.

First, download the training data for NDF objects (~150GB for 3 object classes)

```
cd third_party/ndf_robot
NDF_SOURCE_DIR=$PWD/src/ndf_robot ./scripts/download_training_data.sh
```

Then train embeddings for:
* mug, `cloud_class=0`
* rack, `cloud_class=1`
* gripper, `cloud_class=2`

These embeddings will be used across anything that uses pre-trained embeddings.

```
python scripts/pretrain_embedding.py cloud_class=0
python scripts/pretrain_embedding.py cloud_class=1
python scripts/pretrain_embedding.py cloud_class=2
```

We also provide pre-trained embeddings for the NDF tasks here:
* `trained_models`
    * `pretraining_mug_embnn_weights.ckpt`: pretrained embedding for mug
    * `pretraining_rack_embnn_weights.ckpt`: pretrained embedding for rack
    * `pretraining_gripper_embnn_weights.ckpt`: pretrained embedding for gripper

### Download the training data
Run this command to download pre-generated training data for NDF mug task.

```
bash download_mug_train_data.sh
```

If you want to download pre-generated training data for all three NDF tasks (mug, bottle, bowl), run this instead
```
bash download_all_ndf_train_data.sh
```
### Train models.

```
# Mug, grasp
python scripts/train_residual_flow.py task=mug_grasp

# Mug, place
python scripts/train_residual_flow.py task=mug_place
```

To use custom pre-trained embeddings, add the following flag to the above commands:

```
checkpoint_file_action=<path to action embeddings>
checkpoint_file_anchor=<path to anchor embeddings>
```
If not specified, by default it will uses the provided pre-trained embeddings in `taxpose/trained_models`.

Each of these scripts generates a **model checkpoint** file. You can find the path to the **model checkpoint** file in the output of the script `taxpose/train_new.txt`, under `working_dir: <model checkpoint>`.

### Evaluate models.

To evaluate the models we provided, run the following

```
# Mug, upright
python scripts/evaluate_ndf_mug.py pose_dist=upright

# Mug, arbitrary
python scripts/evaluate_ndf_mug.py pose_dist=arbitrary
```

To use custom-trained models, add the following flags to the above commands:

```
checkpoint_file_grasp=<upright grasp path>
checkpoint_file_place=<upright place path>
```
Substitute `<upright {grasp, place} path>` with the **model checkpoint** you trained above

You can find the evaluation results in the `log_txt_file`, currently defaulted to `taxpose/test_results.txt`.

The success rate for **Grasp**, **Place**, **Overall** as seen in Table 1 as:
```
Iteration: 99, Grasp Success Rate: **Grasp**, Place [teleport] Success Rate: **Place**, overall success Rate: **Overall**
```

## Table 2: NDF # of Demos.

### Train models.

```
# 1 demo
python scripts/train_residual_flow.py task=mug_grasp num_demo=1
python scripts/train_residual_flow.py task=mug_place num_demo=1

# 5 demos
python scripts/train_residual_flow.py task=mug_grasp num_demo=5
python scripts/train_residual_flow.py task=mug_place num_demo=5
```

Each of these scripts generates a **model checkpoint** file. You can find the path to the **model checkpoint** file in the output of the script `taxpose/train_new.txt`, under `working_dir: <model checkpoint>`.

### Run evaluation.

Run evaluation on these models
```
# Mug, upright
python scripts/evaluate_ndf_mug.py pose_dist=upright checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>

# Mug, arbitrary
python scripts/evaluate_ndf_mug.py pose_dist=arbitrary checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>
```
Substitute `<upright {grasp, place} path>` with the **model checkpoint** you trained above with 1/5 demos

You can find the evaluation results in the `log_txt_file`, currently defaulted to `taxpose/test_results.txt`.

The success rate for **Grasp**, **Place**, **Overall** as seen in Table 2 as:
```
Iteration: 99, Grasp Success Rate: **Grasp**, Place [teleport] Success Rate: **Place**, overall success Rate: **Overall**
```

## Table 3: NDF Ablations.

### Train models.

```
# No residuals.
python scripts/train_residual_flow_ablation.py ablation=4_no_residuals task=mug_grasp
python scripts/train_residual_flow_ablation.py ablation=4_no_residuals task=mug_place

# Unweighted SVD.
python scripts/train_residual_flow_ablation.py ablation=5_unweighted_svd task=mug_grasp
python scripts/train_residual_flow_ablation.py ablation=5_unweighted_svd task=mug_place

# No attention.
python scripts/train_residual_flow_ablation.py ablation=8_mlp task=mug_grasp
python scripts/train_residual_flow_ablation.py ablation=8_mlp task=mug_place
```

Each of these scripts generates a **model checkpoint** file. You can find the path to the **model checkpoint** file in the output of the script `taxpose/train_ablation.txt`, under `working_dir: <model checkpoint>`.

### Evaluate.

Run evaluation on these models
```
# Mug, upright
python scripts/evaluate_ndf_mug_ablation.py ablation=<ablation type> pose_dist=upright checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>

# Mug, arbitrary
python scripts/evaluate_ndf_mug_ablation.py ablation=<ablation type> pose_dist=arbitrary checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>
```

Substitute `<ablation type>` with:
* `4_no_residuals` for no residuals
* `5_unweighted_svd` for unweighted SVD
* `8_mlp` for no attention

Substitute `<upright {grasp, place} path>` with the **model checkpoint** you trained above with ablation options

You can find the evaluation results in the `log_txt_file`, currently defaulted to `taxpose/test_results_ablation.txt`.

The success rate for **Grasp**, **Place**, **Overall** as seen in Table 3 as:
```
Iteration: 99, Grasp Success Rate: **Grasp**, Place [teleport] Success Rate: **Place**, overall success Rate: **Overall**
```

## Table 4: PM Placement

### Download the Partnet-Mobility Dataset.

Follow instructions to download here: https://sapien.ucsd.edu/downloads

After downloading, create a dataset directory with the structure:
```
partnet-mobility/
    raw/
        7236/
        ...  # all the partnet-mobility object directories
```

This will become `--pm-root` and `--root` for all the scripts below.

### Train+Evaluate models.

#### TAX-Pose GC

```
# Train

# Evaluate
```

#### TAX-Pose (no-GC)

TODO(beisner): Still need to add the code for this, although its performance is matched/beat by TAX-Pose GC.

#### Baselines

We provide the following dataset for training the baselines methods mentioned in our paper.
[free_floationg_traj_interp_multigoals.zip](https://drive.google.com/file/d/18aLo7PZ9wv58GO2kJxlS8ai8OKE5Pm1l/view?usp=share_link)
In order to create the dataset on your own, please install OMPL on your machine following the instructions [here](https://ompl.kavrakilab.org/installation.html)

Then, you can run the following command:
```python -m taxpose.training.pm_baselones.ompl_traj_gen```

This will create and log the dataset to this directory: ```./taxpose/datasets/pm_data/free_floating_traj_interp_multigoals```

Top reproduce Table 4 in the paper, simply run the provided Jupyter Notebook:

```
./taxpose/training/pm_baselines/baselines_results_agg.ipynb
```


NOTE: This uses wandb to log results and generate checkpoint structure.

**BC**

Train:
```
python taxpose/training/pm_baselines/train.py --model-type bc
```

Evaluate:
```
# In
python taxpose/training/pm_baselines/test_bc.py --cat all --method bc --model <wandb model name>  --postfix 0

# Top
python taxpose/training/pm_baselines/test_bc.py --cat all --method bc --model <wandb model name>  --postfix 1

# Left
python taxpose/training/pm_baselines/test_bc.py --cat all --method bc --model <wandb model name>  --postfix 2

# Right
python taxpose/training/pm_baselines/test_bc.py --cat all --method bc --model <wandb model name>  --postfix 3
```

**DAgger**

Train:
```
python taxpose/training/pm_baselines/train.py --model-type dagger
```

Evaluate:
```
# In
python taxpose/training/pm_baselines/test_bc.py --cat all --method dagger --model <wandb model name>  --postfix 0

# Top
python taxpose/training/pm_baselines/test_bc.py --cat all --method dagger --model <wandb model name>  --postfix 1

# Left
python taxpose/training/pm_baselines/test_bc.py --cat all --method dagger --model <wandb model name>  --postfix 2

# Right
python taxpose/training/pm_baselines/test_bc.py --cat all --method dagger --model <wandb model name>  --postfix 3
```

**TrajFlow**

Train:
```
python taxpose/training/pm_baselines/train.py --model-type traj_flow
```

Evaluate:
```
# In
python taxpose/training/pm_baselines/test_bc.py --cat all --method traj_flow --model <wandb model name>  --postfix 0

# Top
python taxpose/training/pm_baselines/test_bc.py --cat all --method traj_flow --model <wandb model name>  --postfix 1

# Left
python taxpose/training/pm_baselines/test_bc.py --cat all --method traj_flow --model <wandb model name>  --postfix 2

# Right
python taxpose/training/pm_baselines/test_bc.py --cat all --method traj_flow --model <wandb model name>  --postfix 3
```

**GoalFlow**

Train:
```
python taxpose/training/pm_baselines/train.py --model-type goal_flow
```

Evaluate:

NOTE: These are a different file than the BC/Dagger/TrajFlow evaluations because they use a different evaluation script.
```
# In
python taxpose/training/pm_baselines/test_goal_flow.py --cat all --method goal_flow --model <wandb model name>  --postfix 0

# Top
python taxpose/training/pm_baselines/test_goal_flow.py --cat all --method goal_flow --model <wandb model name>  --postfix 1

# Left
python taxpose/training/pm_baselines/test_goal_flow.py --cat all --method goal_flow --model <wandb model name>  --postfix 2

# Right
python taxpose/training/pm_baselines/test_goal_flow.py --cat all --method goal_flow --model <wandb model name>  --postfix 3
```

### Generate Results Table.

## Table 5: Real-world experiments

This is based on a specific real-world experiment we ran, and so can't be reproduced purely from the code in this repository.

## Supplement Table 6: Mug-hanging ablations

### Train Ablation Models

```
# For each <ablation type>, grasp
python scripts/train_residual_flow_ablation.py ablation=<ablation type> task=mug_grasp
# For each <ablation type>, place
python scripts/train_residual_flow_ablation.py ablation=<ablation type> task=mug_place
```

Substitute `<ablation type>` with:
* `0_no_disp_loss`: No L_disp
* `1_no_corr_loss`: No L_corr
* `2_no_cons_loss`: No L_cons
* `3_no_disp_loss_combined`: Scaled Combination: 1.1*L_cons + L_corr
* `4_no_residuals`: No Adjustment via Correspondence Residuals
* `5_unweighted_svd`: Unweighted SVD
* `6_no_finetuning`: No Finetuning for Embedding Network
* `7_no_pretraining`: No Pretraining for Embedding Network
* `8_mlp`: 3-Layer MLP In Place of Transformer
* `9_low_dim_embedding`: Embedding Network Feature Dim = 16

### Evaluate Ablation Models
Run evaluation on these models
```
# Mug, upright
python scripts/evaluate_ndf_mug_ablation.py ablation=<ablation type> pose_dist=upright checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>

# Mug, arbitrary
python scripts/evaluate_ndf_mug_ablation.py ablation=<ablation type> pose_dist=arbitrary checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>
```
Substitute `<ablation type>` with the corresponding ablation type
Substitute `<upright {grasp, place} path>` with the **model checkpoint** you trained above with ablation options

You can find the evaluation results in the `log_txt_file`, currently defaulted to `taxpose/test_results_ablation.txt`.

The success rate for **Grasp**, **Place**, **Overall** as seen in Table 6 as:
```
Iteration: 99, Grasp Success Rate: **Grasp**, Place [teleport] Success Rate: **Place**, overall success Rate: **Overall**
```

## Supplement Table 7: Pre-training ablations


Train models for line 1 trained for 26K steps
```
python scripts/train_residual_flow_ablation.py ablation=7_no_pretraining task=mug_grasp max_epochs=200
python scripts/train_residual_flow_ablation.py ablation=7_no_pretraining task=mug_place max_epochs=200
```

Train models for line 2 trained for 15K steps:
```
python scripts/train_residual_flow_ablation.py ablation=7_no_pretraining task=mug_grasp max_epochs=120
python scripts/train_residual_flow_ablation.py ablation=7_no_pretraining task=mug_place max_epochs=120
```

Evaluate these models
```
# Mug, upright
python scripts/evaluate_ndf_mug_ablation.py ablation=7_no_pretraining pose_dist=upright checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>
```
Substitute `<upright {grasp, place} path>` with the **model checkpoint** you trained above with ablation options

## Supplement Table 8: Additional simulation experiments

If you have only downloaded pre-generated the NDF mug training data, run this to download the training data for bottle and bowl
```
bash download_bottle_bowl_train_data.sh
```

Train models for *bottle*
```
# Bottle, grasp
python scripts/train_residual_flow.py task=bottle_grasp

# Bottle, place
python scripts/train_residual_flow.py task=bottle_place

```

Train models for *bowl*
```
# Bowl, grasp
python scripts/train_residual_flow.py task=bowl_grasp

# Bowl, place
python scripts/train_residual_flow.py task=bowl_place
```

Each of these scripts generates a **model checkpoint** file. You can find the path to the **model checkpoint** file in the output of the script `taxpose/train_new.txt`, under `working_dir: <model checkpoint>`.

Run evaluation on trained *bottle* models
```
# Bottle, upright
python scripts/evaluate_ndf_mug.py pose_dist=upright object_class=bottle checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>

# Bottle, arbitrary
python scripts/evaluate_ndf_mug.py pose_dist=arbitrary object_class=bottle checkpoint_file_grasp=<uparbitraryright grasp path> checkpoint_file_place=<arbitrary place path>
```

Run evaluation on trained *bowl* models
```
# Bowl, upright
python scripts/evaluate_ndf_mug.py pose_dist=upright object_class=bowl checkpoint_file_grasp=<upright grasp path> checkpoint_file_place=<upright place path>

# Bowl, arbitrary
python scripts/evaluate_ndf_mug.py pose_dist=arbitrary object_class=bowl checkpoint_file_grasp=<arbitrary grasp path> checkpoint_file_place=<arbitrary place path>
```

Substitute `<{arbitrary, upright} {grasp, place} path>` with the **model checkpoint** you trained above

## Supplement Table 9: Expanded results

These are granular results of the experiments in Table 1.

## Supplement Table 10: Expanded results

These are real-world experiments we ran, and so can't be reproduced purely from the code in this repository.

## Supplement Tables 11-14: Expanded results

These are granular results of the experiments in Table 1.


## Docker

### Build a Docker container.

```
docker build -t beisner/taxpose .
```

### Run training.

```
WANDB_API_KEY=<API_KEY>
USERNAME=<USERNAME>
# Optional: mount current directory to run / test new code.
# Mount data directory to access data.
docker run \
    --shm-size=256m\
    -v /data/ndf:/opt/baeisner/data \
    -v $(pwd)/trained_models:/opt/baeisner/code/trained_models \
    -v $(pwd)/logs:/opt/baeisner/logs \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_DOCKER_IMAGE=beisner/taxpose \
    beisner/taxpose python scripts/train_residual_flow.py \
        task=mug_grasp \
        model=taxpose \
        +mode=train \
        benchmark.dataset_root=/opt/baeisner/data \
        log_dir=/opt/baeisner/logs
```
