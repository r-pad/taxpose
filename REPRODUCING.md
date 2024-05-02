# Reproducing the paper.

## High-level TODOS

## Pretraining

All tasks require pretraining.

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

TODO: fill in this table.

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
