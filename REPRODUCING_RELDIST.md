# Reproducing the RelDist paper.

## Table 1 - RLBench Placement Tasks

```
####### insert_onto_square_peg ########

# Taxpose
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/insert_onto_square_peg/train_place_taxpose
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/insert_onto_square_peg/eval_place_taxpose

# MLAT
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/insert_onto_square_peg/train_place_mlat
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/insert_onto_square_peg/eval_place_mlat

####### phone_on_base ########

# Taxpose
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/phone_on_base/train_place_taxpose
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/phone_on_base/eval_place_taxpose

# MLAT
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/phone_on_base/train_place_mlat
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/phone_on_base/eval_place_mlat

####### place_hanger_on_rack ########

# Taxpose
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/place_hanger_on_rack/train_place_taxpose
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/place_hanger_on_rack/eval_place_taxpose

# MLAT
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/place_hanger_on_rack/train_place_mlat
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/place_hanger_on_rack/eval_place_mlat

####### put_toilet_roll_on_stand ########

# Taxpose
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/put_toilet_roll_on_stand/train_place_taxpose
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/put_toilet_roll_on_stand/eval_place_taxpose

# MLAT
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/put_toilet_roll_on_stand/train_place_mlat
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/put_toilet_roll_on_stand/eval_place_mlat

####### stack_wine ########

# Taxpose
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/stack_wine/train_place_taxpose
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/stack_wine/eval_place_taxpose

# MLAT
python scripts/train_residual_flow.py --config-name commands/rlbench/reldist_experiments/stack_wine/train_place_mlat
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/stack_wine/eval_place_mlat

```

## Table 2 - Mug Hanging (Upright/Arbitrary)

```
(Optional) Pretraining.
python scripts/pretrain_embeddings.py --config-name commands/ndf/pretraining/mug_vnn
python scripts/pretrain_embeddings.py --config-name commands/ndf/pretraining/rack_vnn
python scripts/pretrain_embeddings.py --config-name commands/ndf/pretraining/gripper_vnn

# Train mug grasp.
python scripts/train_residual_flow.py --config-name commands/ndf/mug/train_grasp_mlat <OPTIONAL PRETRAINING CKPTS>

# Train mug placement.
python scripts/train_residual_flow.py --config-name commands/ndf/mug/train_place_mlat <OPTIONAL PRETRAINING CKPTS>

# Eval mug upright.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_upright_mlat <TRAINED MODEL CKPTS>

# Eval mug arbitrary.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_arbitrary_mlat <TRAINED MODEL CKPTS>
```

## Table 3 - Other NDF Tasks

### Bottle

```
(Optional) Pretraining.
python scripts/pretrain_embeddings.py --config-name commands/ndf/pretraining/bottle_vnn
python scripts/pretrain_embeddings.py --config-name commands/ndf/pretraining/slab_vnn

# Train bottle grasp.
python scripts/train_residual_flow.py --config-name commands/ndf/bottle/train_grasp_mlat <OPTIONAL PRETRAINING CKPTS>

# Train bottle placement.
python scripts/train_residual_flow.py --config-name commands/ndf/bottle/train_place_mlat <OPTIONAL PRETRAINING CKPTS>

# Eval bottle upright.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/bottle/eval_ndf_upright_mlat <TRAINED MODEL CKPTS>

# Eval bottle arbitrary.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/bottle/eval_ndf_arbitrary_mlat <TRAINED MODEL CKPTS>
```

### Bowl

```
(Optional) Pretraining.
python scripts/pretrain_embeddings.py --config-name commands/ndf/pretraining/bowl_vnn
python scripts/pretrain_embeddings.py --config-name commands/ndf/pretraining/slab_vnn

# Train bowl grasp.
python scripts/train_residual_flow.py --config-name commands/ndf/bowl/train_grasp_mlat <OPTIONAL PRETRAINING CKPTS>

# Train bowl placement.
python scripts/train_residual_flow.py --config-name commands/ndf/bowl/train_place_mlat <OPTIONAL PRETRAINING CKPTS>

# Eval bowl upright.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/bowl/eval_ndf_upright_mlat <TRAINED MODEL CKPTS>

# Eval bowl arbitrary.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/bowl/eval_ndf_arbitrary_mlat <TRAINED MODEL CKPTS>
```

### Mug

See Table 1.

## Table 4 - Mug Hanging (z/SE(3))

### MLAT

```
# Train mug grasp.
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/se3_augmentations/train_mug_grasp_mlat <OPTIONAL PRETRAINING CKPTS>

# Train mug placement.
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/se3_augmentations/train_mug_place_mlat <OPTIONAL PRETRAINING CKPTS>

# Eval mug upright.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_upright_mlat <TRAINED MODEL CKPTS>

# Eval mug arbitrary.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_arbitrary_mlat <TRAINED MODEL CKPTS>
```

### TAXPose

```
# Train mug grasp.
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/se3_augmentations/train_mug_grasp <OPTIONAL PRETRAINING CKPTS>

# Train mug placement.
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/se3_augmentations/train_mug_place <OPTIONAL PRETRAINING CKPTS>

# Eval mug upright.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_upright <TRAINED MODEL CKPTS>

# Eval mug arbitrary.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_arbitrary <TRAINED MODEL CKPTS>
```

## Table 5 & 6 - Method ablations

### K=100

```
# Train mug grasp.
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/mlat_smaller_sampling/train_mug_grasp_mlat <OPTIONAL PRETRAINING CKPTS>

# Train mug placement.
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/mlat_smaller_sampling/train_mug_place_mlat <OPTIONAL PRETRAINING CKPTS>

# Eval mug upright.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_upright_mlat <TRAINED MODEL CKPTS>

# Eval mug arbitrary.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_arbitrary_mlat <TRAINED MODEL CKPTS>
```

### No-pretrain

Just do the same mug as above, without optional pretraining checkpoints.

### No Vector Neurons

```
# Train mug grasp.
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/mlat_no_vns/train_mug_grasp_mlat <OPTIONAL PRETRAINING CKPTS>

# Train mug placement.
python scripts/train_residual_flow.py --config-name commands/ndf/ablations/mlat_no_vns/train_mug_place_mlat <OPTIONAL PRETRAINING CKPTS>

# Eval mug upright.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_upright_mlat <TRAINED MODEL CKPTS>

# Eval mug arbitrary.
python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf/mug/eval_ndf_arbitrary_mlat <TRAINED MODEL CKPTS>
```

## Table 7

N/A, real-world reproduction requires our robot setup.

## Figure 7

The sample complexity ablation experiments are done as follows:

```
# Train
python scripts/train_residual_flow.py --configs/commands/rlbench/reldist_experiments/ablations/num_samples/train_place_1_taxpose.yaml
python scripts/train_residual_flow.py --configs/commands/rlbench/reldist_experiments/ablations/num_samples/train_place_5_taxpose.yaml
python scripts/train_residual_flow.py --configs/commands/rlbench/reldist_experiments/ablations/num_samples/train_place_1_mlat.yaml
python scripts/train_residual_flow.py --configs/commands/rlbench/reldist_experiments/ablations/num_samples/train_place_5_mlat.yaml

# Eval
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/ablations/num_samples/eval_place_1_taxpose.yaml
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/ablations/num_samples/eval_place_5_taxpose.yaml
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/ablations/num_samples/eval_place_1_mlat.yaml
python scripts/eval_metrics.py --config-name commands/rlbench/reldist_experiments/ablations/num_samples/eval_place_5_mlat.yaml
```
