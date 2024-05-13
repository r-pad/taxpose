# Reproducing the paper.

In our work, we compare against the performance of Neural Descriptor Fields on the task they released in their paper.

Original code can be found here: https://github.com/anthonysimeonov/ndf_robot

First, download the pretraining data for NDF objects (~150GB for 3 object classes)

```
cd third_party/ndf_robot
NDF_SOURCE_DIR=$PWD/src/ndf_robot ./scripts/download_training_data.sh
```

## Pretraining

All tasks require pretraining.

We  provide pre-trained embeddings for the NDF tasks here:
* `trained_models`
* `pretraining_mug_embnn_weights.ckpt`: pretrained embedding for mug
* `pretraining_rack_embnn_weights.ckpt`: pretrained embedding for rack
* `pretraining_gripper_embnn_weights.ckpt`: pretrained embedding for gripper

### Bottle

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/bottle data_root=/data
```

### Bowl

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/bowl data_root=/data
```

### Gripper

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/gripper data_root=/data
```

### Mug

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/mug data_root=/data
```

### Rack

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/rack data_root=/data
```

### Slab

Note: this one appears broken.

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/slab data_root=/data
```

## Table 1

This table trains the mug on the grasp and place tasks, and evaluates the model on the upright and arbitrary settings of mug-hanging. Reported results are success rates.

Models are optionally logged to WandB, and eval data will be printed and saved to a file.

### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/mug/train_grasp
```

### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/mug/train_place
```

### Evaluate

Upright:

```
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_upright checkpoint_file_grasp=??? checkpoint_file_place=??? seed=??? pybullet_viz=False
```

Arbitrary:

```
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_arbitrary checkpoint_file_grasp=??? checkpoint_file_place=??? seed=??? pybullet_viz=False
```

The success rate for **Grasp**, **Place**, **Overall** as seen in Table 1 as:
```
Iteration: 99, Grasp Success Rate: **Grasp**, Place [teleport] Success Rate: **Place**, overall success Rate: **Overall**
```

## Table 2

This table compares sample-efficiency for {1, 5, 10} demos on the mug-hanging task, and evaluates only on upright setting. Reported results are Overall success rates.

### Train Mug Grasp 1 Demo

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/n_demos/train_mug_grasp_1
```

### Train Mug Place 1 Demo

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/n_demos/train_mug_place_1
```

### Train Mug Grasp 5 Demos

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/n_demos/train_mug_grasp_5
```

### Train Mug Place 5 Demos

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/n_demos/train_mug_place_5
```
### Evaluate

```
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_upright checkpoint_file_grasp=??? checkpoint_file_place=??? seed=??? pybullet_viz=False
```


## Table 3

This table contains ablations. All are trained on 10 demos of mug-hanging, and evaluated in the upright setting.

### No residual

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/4_no_residuals/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/4_no_residuals/train_mug_place
```

#### Evaluate

TODO.

### Unweighted SVD

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/5_unweighted_svd/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/5_unweighted_svd/train_mug_place
```

#### Evaluate

TODO.

### No Cross-Attention

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/8_mlp/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/8_mlp/train_mug_place
```

#### Evaluate

TODO.

## Table 4

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

## Table 5 - Attention weight ablation

Mug hanging, upright.

TODO: Not sure what this was...

### Train Mug Grasp

### Train Mug Place

### Evaluate

## Table 6

### No L_disp

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/0_no_disp_loss/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/0_no_disp_loss/train_mug_place
```

#### Evaluate

TODO.

### No L_corr

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/1_no_corr_loss/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/1_no_corr_loss/train_mug_place
```

#### Evaluate

TODO.

### No L_cons

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/2_no_cons_loss/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/2_no_cons_loss/train_mug_place
```

#### Evaluate

TODO

### Scaled loss combo 1.1 * L_cons + L_corr

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/3_no_disp_loss_combined/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/3_no_disp_loss_combined/train_mug_place
```

#### Evaluate

TODO.

### No correspondence residuals.

See above.

### Unweighted SVD

See above.

### No finetuning of embedding network

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/6_no_finetuning/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/6_no_finetuning/train_mug_place
```

#### Evaluate

TODO.

### No pretraining of embedding network

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/7_no_pretraining/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/7_no_pretraining/train_mug_place
```

#### Evaluate

### 3-layer MLP instead of Transformer

See above.

### Embedding network feature dim = 16

#### Train Mug Grasp

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/9_low_dim_embedding/train_mug_grasp
```

#### Train Mug Place

```
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/9_low_dim_embedding/train_mug_place
```

#### Evaluate

TODO.

## Table 7 - Pretraining

TODO: fill in this table.

## Table 8 - Bottle & Bowl

### Train Bottle Grasp

Broken: sampling is not working correctly.

```
python scripts/train_residual_flow.py --config-name commands/ndf/bottle/train_grasp
```

### Train Bottle Place

Broken: sampling is not working correctly.

```
python scripts/train_residual_flow.py --config-name commands/ndf/bottle/train_place
```

### Evaluate Bottle

### Train Bowl Grasp

Broken: sampling is not working correctly.

```
python scripts/train_residual_flow.py --config-name commands/ndf/bowl/train_grasp
```

### Train Bowl Place

Broken: sampling is not working correctly.

```
python scripts/train_residual_flow.py --config-name commands/ndf/bowl/train_place
```

### Evaluate Bowl

## Table 9 - PM Placement

These are granular results of the experiments in Table 4.

## Table 10 - PM Placement

These are real-world experiments we ran, and so can't be reproduced purely from the code in this repository.

## Table 11 - PM Placement

These are granular results of the experiments in Table 4.

## Table 12 - PM Placement

These are granular results of the experiments in Table 4.

## Table 13 - PM Placement

These are granular results of the experiments in Table 4.

## Table 14 - PM Placement

These are granular results of the experiments in Table 4.
