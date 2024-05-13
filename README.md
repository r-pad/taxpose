# TAX-Pose: Methods for Relative Placement

This is the official code release for our CoRL 2022 paper and ICLR 2024 paper:

**TAX-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation**
*Chuer Pan\*, Brian Okorn\*, Harry Zhang\*, Ben Eisner\*, David Held*
CoRL 2022
[website](https://sites.google.com/view/tax-pose/home) | [paper](https://openreview.net/forum?id=YmJi0bTfeNX)

```
@inproceedings{pan2022taxpose,
    title       = {{TAX}-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation},
    author      = {Chuer Pan and Brian Okorn and Harry Zhang and Ben Eisner and David Held},
    booktitle   = {6th Annual Conference on Robot Learning},
    year        = {2022},
    url         = {https://openreview.net/forum?id=YmJi0bTfeNX}
}
```

Commands to reproduce the results in the paper are detailed in [REPRODUCING_TAXPOSE.md](REPRODUCING_TAXPOSE.md).

**Deep SE(3)-Equivariant Geometric Reasoning for Precise Placement Tasks**
Ben Eisner, Yi Yang, Todor Davchev, Mel Vecerik, Jon Scholz, David Held
ICLR 2024
[website](https://sites.google.com/view/reldist-iclr-2023) | [paper](https://openreview.net/forum?id=2inBuwTyL2)

```
@inproceedings{eisner2024deep,
    title       = {Deep {SE}(3)-Equivariant Geometric Reasoning for Precise Placement Tasks},
    author      = {Ben Eisner and Yi Yang and Todor Davchev and Mel Vecerik and Jonathan Scholz and David Held},
    booktitle   = {The Twelfth International Conference on Learning Representations},
    year        = {2024},
    url         = {https://openreview.net/forum?id=2inBuwTyL2}
}
```

Commands to reproduce the results in the paper are detailed in [REPRODUCING_RELDIST.md](REPRODUCING_RELDIST.md).

Questions? Open an Issue, or send an email to:
ben [dot] a [dot] eisner [at] gmail [dot] com


# ⚠️ Reproduciblilty for NDF Tasks

Some of the NDF eval scripts are not bit-for-bit reproducible even with random seeds. Some of the motion planning in the NDF repository depends on time elapsed, instead of number of steps. So predictions seem to be quite reproducible, but the exact motion planning trajectories are not (meaning that teleport scores are deterministic, but not the executed ones). This means that the NDF results in the original TAX-Pose paper and NDF paper are noisy, and don't yield the exact same results on every run as published in the paper.

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

We've provided specific GPU versions for each of these repositories in `requirements-gpu.txt`. Install them with:

```
pip install -r requirements-gpu.txt
```


# NOTE
Cursed vglrun requires us to pipe in our own custom LD_LIBRARY_PATH. This is a hacky way to do it, but it works for now.

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

# I want to...

## ...use the TAX-Pose model code.

Things are a bit messy at the moment, but the main entrypoint for the model code is in `transformer_flow.py`. The `ResidualFlow_DiffEmbTransformer` class is the main TAX-Pose model, and can be configured in various ways to add different features / parameters.

TODO(beisner): Add an example.

TODO(beisner): fix up the model code to be more user-friendly, including predicting a rigid transform instead of in the "model" code.

## ...use the RelDist / multilateration code.

See:
* `taxpose/utils/multilateration.py`: The differentiable multilateration code.
* `taxpose/nets/transformer_flow.py`: The `MultilaterationHead` class and the `MLPKernel` class are the main components of the multilateration code. They are used as drop-in replacements the TAX-Pose model.

## ...use the data / dataset code used in these papers.

TODO(beisner): Add instructions.

### NDF raw data

### Our post-processed NDF datasets (mug, bowl, bottle)

### Partnet-Mobility

### RLBench

## ...use a pre-trained TAX-Pose model.

TODO(beisner): Add the pretrained models.

### NDF Mug.

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


# Code Structure

TODO(beisner): Clean this up.

TODO(beisner): Clean up the code structure a little bit, one day.

* `configs/`: Hydra configuration files for training and evaluation, for all the datasets.
    * `benchmark`: ndf, rlbench, or real_world
    * `commands`: Specific commands for training and evaluation.
    * `datasets`: Specific datasets for each task.
    * `model`: Different model configurations (taxpose, mlat).
        * `encoder`: Different encoder configurations (vn_dgcnn, dgcnn).
    * `object_class`: Different object classes (mug, bowl, bottle). Used for pretraining.
    * `pose_dist`: Different pose distributions (upright, arbitrary).
    * `task`: Different tasks (ndf tasks, rlbench tasks), broken down into various phases. The RLBench phases are autogenerated.
* `docs/`: (eventual) autogenerated documentation for the code
* `notebooks/`
    * `pm_placement.ipynb`: Simple visualization of the PM Placement dataset.
* `results/`: The pm evaluation script will dump CSVs of the results here.
* `scripts/`
    * `create_pm_dataset.py`: Script which will generate the cached version Partnet-Mobility Placement dataset.
    * `eval_metrics.py`: Evaluate precision metrics on a dataset.
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
