# Scripts for your project

If you write some scripts which are meant to be run stand-alone, and not imported as part of the library, put them in this directory.


## Full pretraining.

### Vector neurons.
python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_bottle wandb.group=pretrain_vn
python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_bowl wandb.group=pretrain_vn
python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_gripper wandb.group=pretrain_vn
python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_mug wandb.group=pretrain_vn
python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_rack wandb.group=pretrain_vn
python scripts/pretrain_embedding.py --config-name=commands/pretrain_vn_dgcnn_table wandb.group=pretrain_vn

### Normal DGCNN.
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_bottle wandb.group=pretrain
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_bowl wandb.group=pretrain
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_gripper wandb.group=pretrain
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_mug wandb.group=pretrain
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_rack wandb.group=pretrain
python scripts/pretrain_embedding.py --config-name=commands/pretrain_dgcnn_table wandb.group=pretrain
