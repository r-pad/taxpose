import os

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from taxpose.datasets.pretraining_point_cloud_data_module import (
    PretrainingMultiviewDataModule,
)
from taxpose.nets.transformer_flow import EquivariantFeatureEmbeddingNetwork
from taxpose.training.equivariant_feature_pretraining_module import (
    EquivariancePreTrainingModule,
)
from taxpose.utils.callbacks import SaverCallbackEmbnn

# chuerp conda env: pytorch3d_38


@hydra.main(config_path="../configs", config_name="pretraining_mug_dgcnn")
def main(cfg):
    pl.seed_everything(cfg.seed)
    logger = WandbLogger(project=cfg.experiment)
    logger.log_hyperparams(cfg)
    logger.log_hyperparams({"working_dir": os.getcwd()})
    trainer = pl.Trainer(
        logger=logger,
        gpus=1,
        reload_dataloaders_every_n_epochs=1,
        # val_check_interval=0.2,
        # val_check_interval=10,
        check_val_every_n_epoch=10,
        callbacks=[SaverCallbackEmbnn()],
    )
    dm = PretrainingMultiviewDataModule(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cloud_class=cfg.cloud_class,
        dataset_root=cfg.dataset_root,
        dataset_index=cfg.dataset_index,
        cloud_type=cfg.cloud_type,
        # overfit=cfg.overfit,
        pretraining_data_path=cfg.pretraining_data_path,
    )

    dm.setup()
    network = EquivariantFeatureEmbeddingNetwork(
        emb_dims=cfg.emb_dims, emb_nn=cfg.emb_nn
    )
    model = EquivariancePreTrainingModule(
        network,
        lr=cfg.lr,
        image_log_period=cfg.image_logging_period,
        l2_reg_weight=cfg.l2_reg_weight,
        normalize_features=cfg.normalize_features,
        temperature=cfg.temperature,
        con_weighting=cfg.con_weighting,
    )
    model.cuda()
    model.train()
    logger.watch(network)

    if cfg.checkpoint_file is not None:
        model.load_state_dict(torch.load(cfg.checkpoint_file)["state_dict"])
    trainer.fit(model, dm)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
