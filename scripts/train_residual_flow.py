import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.nets.multimodal_transformer_flow import (
    Multimodal_ResidualFlow_DiffEmbTransformer, 
    Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX
)
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
from taxpose.training.multimodal_flow_equivariance_training_module_nocentering import (
    Multimodal_EquivarianceTrainingModule,
    Multimodal_EquivarianceTrainingModule_WithPZCondX
)

def write_to_file(file_name, string):
    with open(file_name, "a") as f:
        f.writelines(string)
        f.write("\n")
    f.close()


def load_emb_weights(checkpoint_reference, wandb_cfg=None, run=None):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)
        if run is None or not isinstance(run, wandb.sdk.wandb_run.Run):
            # Download without a run
            api = wandb.Api()
            artifact = api.artifact(checkpoint_reference, type="model")
        else:
            artifact = run.use_artifact(checkpoint_reference)
        checkpoint_path = artifact.get_path("model.ckpt").download(root=artifact_dir)
        weights = torch.load(checkpoint_path)["state_dict"]
        # remove "model.emb_nn" prefix from keys
        weights = {k.replace("model.emb_nn.", ""): v for k, v in weights.items()}
        return weights
    else:
        return torch.load(hydra.utils.to_absolute_path(checkpoint_reference))[
            "embnn_state_dict"
        ]


def maybe_load_from_wandb(checkpoint_reference, wandb_cfg=None, run=None):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        # download checkpoint locally (if not already cached)
        artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)
        if run is None or not isinstance(run, wandb.sdk.wandb_run.Run):
            # Download without a run
            api = wandb.Api()
            artifact = api.artifact(checkpoint_reference, type="model")
        else:
            artifact = run.use_artifact(checkpoint_reference)
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    return ckpt_file


@hydra.main(version_base="1.1", config_path="../configs", config_name="train_ndf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg.seed)
    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        save_dir=cfg.wandb.save_dir,
        job_type=cfg.job_type,
        save_code=True,
        log_model=True,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
    )
    # logger.log_hyperparams(cfg)
    # logger.log_hyperparams({"working_dir": os.getcwd()})
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        # reload_dataloaders_every_n_epochs=1,
        # callbacks=[SaverCallbackModel(), SaverCallbackEmbnnActionAnchor()],
        callbacks=[
            # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
            # It saves everything, and you can load by referencing last.ckpt.
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}",
                monitor="step",
                mode="max",
                save_weights_only=False,
                save_last=True,
                every_n_epochs=1,
            ),
            # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}-{train_loss:.2f}-weights-only",
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
            ),
        ],
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps,
        gradient_clip_val=cfg.training.gradient_clipping
    )

    dm = MultiviewDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,
        cfg=cfg.dm,
    )

    dm.setup()

    if cfg.model.name in ["taxposed"]:
        TP_input_dims = Multimodal_ResidualFlow_DiffEmbTransformer.TP_INPUT_DIMS[cfg.model.conditioning]
        
        taxpose_decoder_network = ResidualFlow_DiffEmbTransformer(
            emb_dims=cfg.model.emb_dims,
            input_dims=TP_input_dims,
            emb_nn=cfg.model.emb_nn,
            return_flow_component=True,
            center_feature=cfg.model.center_feature,
            pred_weight=cfg.model.pred_weight,
            multilaterate=cfg.model.multilaterate,
            sample=cfg.model.mlat_sample,
            mlat_nkps=cfg.model.mlat_nkps,
            break_symmetry=cfg.break_symmetry,
            conditioning_size=cfg.model.latent_z_linear_size if cfg.model.conditioning in ["latent_z_linear_internalcond"] else 0,
        )
        
        pzY_prediction_network = Multimodal_ResidualFlow_DiffEmbTransformer(
            residualflow_diffembtransformer=taxpose_decoder_network,
            gumbel_temp=cfg.model.gumbel_temp,
            freeze_residual_flow=cfg.training.freeze_residual_flow,
            center_feature=cfg.model.center_feature,
            freeze_z_embnn=cfg.training.freeze_z_embnn,
            division_smooth_factor=cfg.model.division_smooth_factor,
            add_smooth_factor=cfg.model.add_smooth_factor,
            conditioning=cfg.model.conditioning,
            latent_z_linear_size=cfg.model.latent_z_linear_size,
            taxpose_centering=cfg.model.taxpose_centering
        )
        
        training_model = Multimodal_EquivarianceTrainingModule(
            pzY_prediction_network,
            lr=cfg.training.lr,
            image_log_period=cfg.training.image_logging_period,
            point_loss_type=cfg.training.point_loss_type,
            return_flow_component=True,
            rotation_weight=cfg.training.rotation_weight,
            weight_normalize=cfg.task.phase.weight_normalize,
            consistency_weight=cfg.training.consistency_weight,
            smoothness_weight=cfg.training.smoothness_weight,
            action_weight=cfg.training.action_weight,
            vae_reg_loss_weight=cfg.training.vae_reg_loss_weight,
            sigmoid_on=cfg.training.sigmoid_on,
            softmax_temperature=cfg.task.phase.softmax_temperature,
            min_err_across_racks_debug=cfg.training.min_err_across_racks_debug,
            error_mode_2rack=cfg.training.error_mode_2rack
        )
        
        if not cfg.training.joint_train_prior and cfg.training.init_cond_x and (not cfg.training.freeze_embnn or not cfg.training.freeze_residual_flow):
            raise ValueError("YOU PROBABLY DIDN'T MEAN TO DO JOINT TRAINING")
        if not cfg.training.joint_train_prior and cfg.training.init_cond_x and cfg.training.checkpoint_file is None:
            raise ValueError("YOU PROBABLY DIDN'T MEAN TO TRAIN BOTH P(Z|X) AND P(Z|Y) FROM SCRATCH")
    
        if cfg.training.init_cond_x: 
            print(f'--------------- Initializing P(z|X) ---------------')
            pzX_prediction_network = Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(
                residualflow_embnn=pzY_prediction_network,
                encoder_type=cfg.model.pzcondx_encoder_type,
                shuffle_for_pzX=cfg.model.shuffle_for_pzX,
            )
            
            training_model_cond_x = Multimodal_EquivarianceTrainingModule_WithPZCondX(
                pzX_prediction_network,
                training_model,
                goal_emb_cond_x_loss_weight=cfg.training.goal_emb_cond_x_loss_weight,
                joint_train_prior=cfg.training.joint_train_prior,
                joint_train_prior_freeze_embnn=cfg.training.joint_train_prior_freeze_embnn,
                freeze_residual_flow=cfg.training.freeze_residual_flow,
                freeze_z_embnn=cfg.training.freeze_z_embnn,
                freeze_embnn=cfg.training.freeze_embnn)
        
        # TODO: Add support for loading pretraining
        if cfg.training.load_from_checkpoint:
            assert cfg.training.checkpoint_file is not None
            print(f'--------------- Loading Checkpoint File From: ---------------')
            print(cfg.training.checkpoint_file)
            if not cfg.training.load_cond_x:
                training_model.load_state_dict(
                    torch.load(hydra.utils.to_absolute_path(
                        maybe_load_from_wandb(cfg.training.checkpoint_file, cfg.wandb, logger.experiment)))["state_dict"])
                print(f'--------------- Loaded P(z|Y) ---------------')
            else:
                training_model_cond_x.load_state_dict(
                    torch.load(hydra.utils.to_absolute_path(
                        maybe_load_from_wandb(cfg.training.checkpoint_file, cfg.wandb, logger.experiment)))["state_dict"])
                print(f'--------------- Loaded P(z|X) ---------------')
            print(f'--------------- Checkpoint Loaded ---------------')
        
        model = training_model_cond_x if cfg.training.init_cond_x else training_model
        model.cuda()
        model.train()
        
    else:
        network = ResidualFlow_DiffEmbTransformer(
            emb_dims=cfg.model.emb_dims,
            emb_nn=cfg.model.emb_nn,
            return_flow_component=cfg.model.return_flow_component,
            center_feature=cfg.model.center_feature,
            pred_weight=cfg.model.pred_weight,
            multilaterate=cfg.model.multilaterate,
            sample=cfg.model.mlat_sample,
            mlat_nkps=cfg.model.mlat_nkps,
            break_symmetry=cfg.break_symmetry,
        )

        model = EquivarianceTrainingModule(
            network,
            lr=cfg.training.lr,
            image_log_period=cfg.training.image_logging_period,
            displace_loss_weight=cfg.training.displace_loss_weight,
            consistency_loss_weight=cfg.training.consistency_loss_weight,
            direct_correspondence_loss_weight=cfg.training.direct_correspondence_loss_weight,
            weight_normalize=cfg.task.phase.weight_normalize,
            sigmoid_on=cfg.training.sigmoid_on,
            softmax_temperature=cfg.task.phase.softmax_temperature,
            flow_supervision=cfg.training.flow_supervision,
        )

        model.cuda()
        model.train()
        if cfg.training.load_from_checkpoint:
            print("loaded checkpoint from")
            print(cfg.training.checkpoint_file)
            model.load_state_dict(
                torch.load(hydra.utils.to_absolute_path(cfg.training.checkpoint_file))[
                    "state_dict"
                ]
            )

        else:
            # Might be empty and not have those keys defined.
            # TODO: move this pretraining into the model itself.
            # TODO: figure out if we can get rid of the dictionary and make it null.
            if cfg.model.pretraining:
                if cfg.model.pretraining.checkpoint_file_action is not None:
                    # # Check to see if it's a wandb checkpoint.
                    # TODO: need to retrain a few things... checkpoint didn't stick...
                    emb_nn_action_state_dict = load_emb_weights(
                        cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment
                    )
                    # checkpoint_file_fn = maybe_load_from_wandb(
                    #     cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment.run
                    # )

                    model.model.emb_nn_action.load_state_dict(emb_nn_action_state_dict)
                    print(
                        "-----------------------Pretrained EmbNN Action Model Loaded!-----------------------"
                    )
                    print(
                        "Loaded Pretrained EmbNN Action: {}".format(
                            cfg.pretraining.checkpoint_file_action
                        )
                    )
                if cfg.pretraining.checkpoint_file_anchor is not None:
                    emb_nn_anchor_state_dict = load_emb_weights(
                        cfg.pretraining.checkpoint_file_anchor, cfg.wandb, logger.experiment
                    )
                    model.model.emb_nn_anchor.load_state_dict(emb_nn_anchor_state_dict)
                    print(
                        "-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------"
                    )
                    print(
                        "Loaded Pretrained EmbNN Anchor: {}".format(
                            cfg.pretraining.checkpoint_file_anchor
                        )
                    )
    
    trainer.fit(model, dm)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
