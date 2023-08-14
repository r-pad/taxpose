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
./launch_autobot.sh 0 python scripts/train_residual_flow.py --config-name commands/train_taxpose_mug_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 1 python scripts/train_residual_flow.py --config-name commands/train_taxpose_mug_place +mode=train wandb.group=taxpose
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bottle_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 3 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bottle_place +mode=train wandb.group=taxpose
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bowl_grasp +mode=train wandb.group=taxpose
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/train_taxpose_bowl_place +mode=train wandb.group=taxpose

### Multilateration.
./launch_autobot.sh 6 python scripts/train_residual_flow.py --config-name commands/train_mlat_mug_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 7 python scripts/train_residual_flow.py --config-name commands/train_mlat_mug_place +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 2 python scripts/train_residual_flow.py --config-name commands/train_mlat_bottle_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 3 python scripts/train_residual_flow.py --config-name commands/train_mlat_bottle_place +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 4 python scripts/train_residual_flow.py --config-name commands/train_mlat_bowl_grasp +mode=train wandb.group=mlat_s256_vnn
./launch_autobot.sh 5 python scripts/train_residual_flow.py --config-name commands/train_mlat_bowl_place +mode=train wandb.group=mlat_s256_vnn
