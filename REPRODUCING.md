# Reproducing the paper.

## High-level TODOS
[] Create pretraining configs
[] Create ablation configs

## Pretraining

All tasks require pretraining.

### Bottle

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/bottle training.dataset.pretraining_data_path=/data/ndf_original/data data_root=/data/ndf
```

### Bowl

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/bowl training.dataset.pretraining_data_path=/data/ndf_original/data data_root=/data/ndf
```

### Gripper

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/gripper training.dataset.pretraining_data_path=/data/ndf_original/data data_root=/data/ndf
```

### Mug

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/mug training.dataset.pretraining_data_path=/data/ndf_original/data data_root=/data/ndf
```

### Rack

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/rack training.dataset.pretraining_data_path=/data/ndf_original/data data_root=/data/ndf
```

### Slab

Note: this one appears broken.

```
python scripts/pretrain_embedding.py --config-name commands/ndf/pretraining/slab training.dataset.pretraining_data_path=/data/ndf_original/data data_root=/data/ndf
```

## Table 1

This table trains the mug on the grasp and place tasks, and evaluates the model on the upright and arbitrary settings of mug-hanging. Reported results are success rates.

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

## Table 2

This table compares sample-efficiency for {1, 5, 10} demos on the mug-hanging task, and evaluates only on upright setting. Reported results are Overall success rates.

### Train Mug Grasp 1 Demo

TODO: write command.

### Train Mug Grasp 5 Demos

TODO: write command.

### Evaluate

TODO: write command.

## Table 3

This table contains ablations. All are trained on 10 demos of mug-hanging, and evaluated in the upright setting.

### No residual

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

### Unweighted SVD

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

### No Cross-Attention

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

## Table 4

TODO: fill in this table.

## Table 5 - Attention weight ablation

Mug hanging, upright.

### Train Mug Grasp

### Train Mug Place

### Evaluate

## Table 6

### No L_disp

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

### No L_corr

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

### No L_cons

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

### Scaled loss combo 1.1 * L_cons + L_corr

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

### No correspondence residuals.

See above.

### Unweighted SVD

See above.

### No finetuning of embedding network

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

### No pretraining of embedding network

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

### 3-layer MLP instead of Transformer

See above.

### Embedding network feature dim = 16

#### Train Mug Grasp

#### Train Mug Place

#### Evaluate

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

TODO: fill in this table.

## Table 10 - PM Placement

TODO: fill in this table.

## Table 11 - PM Placement

TODO: fill in this table.

## Table 12 - PM Placement

TODO: fill in this table.

## Table 13 - PM Placement

TODO: fill in this table.

## Table 14 - PM Placement

TODO: fill in this table.
