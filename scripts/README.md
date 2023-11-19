# Scripts for your project

If you write some scripts which are meant to be run stand-alone, and not imported as part of the library, put them in this directory.


## Full pretraining.

### Vector neurons.
./launch_autobot.sh 0 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_bottle wandb.group=pretrain_vn
./launch_autobot.sh 1 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_bowl wandb.group=pretrain_vn
./launch_autobot.sh 2 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_gripper wandb.group=pretrain_vn
./launch_autobot.sh 3 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_mug wandb.group=pretrain_vn
./launch_autobot.sh 4 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_rack wandb.group=pretrain_vn
./launch_autobot.sh 5 python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_table wandb.group=pretrain_vn

### Normal DGCNN.
./launch_autobot.sh 0 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_gripper wandb.group=pretrain
./launch_autobot.sh 1 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_mug wandb.group=pretrain
./launch_autobot.sh 2 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_rack wandb.group=pretrain
./launch_autobot.sh 3 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_table wandb.group=pretrain
./launch_autobot.sh 4 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_bowl wandb.group=pretrain
./launch_autobot.sh 5 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_bottle wandb.group=pretrain


## Full training.

### Taxpose.
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/train_taxpose_mug_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/train_taxpose_mug_place +mode=train wandb.group=taxpose
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bottle_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bottle_place +mode=train wandb.group=taxpose
./launch_autobot.sh 6 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bowl_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 7 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bowl_place +mode=train wandb.group=taxpose

### Multilateration.
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/train_mlat_mug_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/train_mlat_mug_place +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/train_mlat_bottle_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/train_mlat_bottle_place +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 6 python scripts/train_residual_flow.py --config-name commands/train_mlat_bowl_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 7 python scripts/train_residual_flow.py --config-name commands/train_mlat_bowl_place +mode=train wandb.group=mlat_s256_vnn


## Ablations

### SE(3)

#### TAX-pose

./launch_autobot.sh 0 python scripts/train_residual_flow.py --config-name commands/ablations/se3_augmentation/train_taxpose_mug_grasp.yaml +mode=train wandb.group=ablate_se3_taxpose

./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/ablations/se3_augmentation/train_taxpose_mug_place.yaml +mode=train wandb.group=ablate_se3_taxpose


####

./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/ablations/se3_augmentation/train_mlat_mug_grasp.yaml +mode=train wandb.group=ablate_se3_mlat_s256_vnn

./launch_autobot.sh 3 python scripts/train_residual_flow.py --config-name commands/ablations/se3_augmentation/train_mlat_mug_place.yaml +mode=train wandb.group=ablate_se3_mlat_s256_vnn


## RLBench

### Taxpose

./launch_autobot.sh 0 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_stack_wine_place.yaml wandb.group=rlbench_taxpose
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_insert_onto_square_peg_place.yaml wandb.group=rlbench_taxpose resources.num_workers=0
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_phone_on_base_place.yaml wandb.group=rlbench_taxpose
./launch_autobot.sh 3 python scripts/train_residual_flow.py --config-name commands/rlbench/train_taxpose_rlbench_put_toilet_roll_on_stand_place.yaml wandb.group=rlbench_taxpose
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_stack_wine_place.yaml wandb.group=rlbench_mlat
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_insert_onto_square_peg_place.yaml wandb.group=rlbench_mlat
./launch_autobot.sh 6 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_phone_on_base_place.yaml wandb.group=rlbench_mlat
./launch_autobot.sh 7 python scripts/train_residual_flow.py --config-name commands/rlbench/train_mlat_rlbench_put_toilet_roll_on_stand_place.yaml wandb.group=rlbench_mlat


## NDF evals.
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_mug_upright seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_mug_arbitrary seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_bottle_upright seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_bottle_arbitrary seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_bowl_upright seed=10 wandb.group=taxpose
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/taxpose_bowl_arbitrary seed=10 wandb.group=taxpose

CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_mug_upright seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_mug_arbitrary seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_bottle_upright seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_bottle_arbitrary seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_bowl_upright seed=10 wandb.group=mlat_s256_vnn
CUDA_VISIBLE_DEVICES=1 DISPLAY=:0.0 python scripts/evaluate_ndf_mug_standalone.py --config-name commands/ndf_evals/mlat_bowl_arbitrary seed=10 wandb.group=mlat_s256_vnn
