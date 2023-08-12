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
./launch_autobot.sh 0 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_bottle wandb.group=pretrain
./launch_autobot.sh 1 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_bowl wandb.group=pretrain
./launch_autobot.sh 2 python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_gripper wandb.group=pretrain
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_mug wandb.group=pretrain
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_rack wandb.group=pretrain
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_table wandb.group=pretrain
