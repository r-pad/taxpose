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
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### `pytorch-geometric`

You can follow instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

For our experiments, we installed with the following command:

```
pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 pyg_lib==0.1.0 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

### `pytorch3d`

Follow instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#3-install-wheels-for-linux).

We ran the following:

```
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
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

<!--

## THIS SEEMS DUPLICATED...
### Download the data.

Clone the [ndf_repo](https://github.com/anthonysimeonov/ndf_robot), and follow their instructions for:
* Setup Franka Panda inverse kinematics tools with ikfast
```
cd third_party/ndf_robot/pybullet-planning/pybullet_tools/ikfast/franka_panda
python setup.py
```

* Setup ndf environment variables
```
cd third_party/ndf_robot
source ndf_env.sh
```

* downloading the full dataset

```
cd third_party/ndf_robot
./scripts/download_obj_data.sh
./scripts/download_demo_demonstrations.sh
```
-->

# Code Structure

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

TODO: Fill this in.

## ...use the TAX-Pose model code.

## ...use a pre-trained TAX-Pose model.

Here are links to pre-trained models:

* NDF-Mug
* NDF-Bowl
* NDF-Bottle
* PM-Placement

# How to reproduce all the tables from our paper

## Table 1: NDF Tasks

In our work, we compare against the performance of Neural Descriptor Fields on the task they released in their paper.

Original code can be found here: https://github.com/anthonysimeonov/ndf_robot

### Pretrain the embeddings.

These embeddings will be used across anything that uses pre-trained embeddings.

```
TODO(chuer): Add this.
```

We provide pre-trained embeddings for the NDF tasks here: TODO(chuer): Add links (inside one big folder).

### Train models.

```
# Mug, upright, grasp
python scripts/train_residual_flow.py task=mug_grasp

# Mug, upright, place
python scripts/train_residual_flow.py task=mug_place

# Mug, arbitrary, grasp
TODO(chuer): Add this.

# Mug, arbitrary, place
TODO(chuer): Add this.
```

To use custom pre-trained embeddings, add the following flag to the above commands:

```
checkpoint_file_action=<path to action embeddings>
checkpoint_file_anchor=<path to anchor embeddings>
```

Each of these scripts generates a checkpoint file. You can find the path to the checkpoint file in the output of the script.

### Evaluate models.

```
# Mug, upright
python scripts/evaluate_ndf_mug.py

# Mug, arbitrary
TODO(chuer): Add this.
```

To use custom-trained models, add the following flags to the above commands:

```
checkpoint_file_grasp=<upright grasp path>
checkpoint_file_place=<upright place path>
```

TODO(chuer): Write instructions on how to actually generate the line of Table 1...


## Table 2: NDF # of Demos.

### Train models.

```
# 1 demo
TODO(chuer): Add upright grasp.
TODO(chuer): Add upright place.

# 5 demos
TODO(chuer): Add upright grasp.
TODO(chuer): Add upright place.
```

### Run evaluation.

```
TODO(chuer): Add 1 demo eval.
TODO(chuer): Add 5 demo eval.
```

TODO(chuer): Write instructions on how to actually generate the lines of Table 2...


## Table 3: NDF Ablations.

### Train models.

```
# No residuals.
TODO(chuer): Add upright grasp.
TODO(chuer): Add upright place.

# Unweighted SVD.
TODO(chuer): Add upright grasp.
TODO(chuer): Add upright place.

# No attention.
TODO(chuer): Add upright grasp.
TODO(chuer): Add upright place.
```

### Evaluate.

```
TODO(chuer): Add no residuals eval.
TODO(chuer): Add unweighted SVD eval.
TODO(chuer): Add no attention eval.
```

TODO(chuer): Write instructions on how to actually generate the lines of Table 3...

## Table 4: PM Placement

### Download the Partnet-Mobility Dataset.

### Generate (or download) the cached version of the dataset.

### Train+Evaluate models.

#### TAX-Pose GC

```
# Train

# Evaluate
```

#### TAX-Pose (no-GC)

TODO: This is a bit hacky...

```
# Train

# Evaluate
```

#### E2E BC

TODO

```
# Train

# Evaluate
```


#### E2E DAgger

TODO

```
# Train

# Evaluate
```

#### Traj Flow

TODO

```
# Train

# Evaluate
```

#### Goal Flow

TODO

```
# Train

# Evaluate
```

### Generate Results Table.

## Table 5: Real-world experiments

This is based on a specific real-world experiment we ran, and so can't be reproduced purely from the code in this repository.

## Supplement Table 6: Mug-hanging ablations

TODO

## Supplement Table 7: Pre-training ablations

## Supplement Table 8: Additional simulation experiments

See the results from Table 1.

## Supplement Table 9: Expanded results

TODO

## Supplement Table 10: Expanded results

## Supplement Tables 11-14: Expanded results

See the results from Table 4.
