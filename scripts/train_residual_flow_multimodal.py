# from equivariant_pose_graph.dataset.rpdiff_data_module import RpDiffDataModule
import os

import torch
import torch.multiprocessing

from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.nets.taxposed.multimodal_transformer_flow import (
    Multimodal_ResidualFlow_DiffEmbTransformer,
    Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX,
)
from taxpose.nets.taxposed.transformer_flow import (
    AlignedFrameDecoder,
    ResidualFlow_DiffEmbTransformer,
)
from taxpose.training.taxposed.flow_equivariance_training_module_nocentering_multimodal import (
    EquivarianceTrainingModule,
    EquivarianceTrainingModule_WithPZCondX,
)

torch.multiprocessing.set_sharing_strategy("file_system")

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# chuerp conda env: pytorch3d_38


def setup_main(cfg):
    pl.seed_everything(cfg.seed)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # torch.set_float32_matmul_precision("medium")
    TESTING = "PYTEST_CURRENT_TEST" in os.environ

    pl.seed_everything(cfg.seed)
    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        save_dir=cfg.wandb.save_dir,
        job_type=cfg.job_type,
        save_code=True,
        log_model=True,
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
    )
    # logger.log_hyperparams(cfg)
    # logger.log_hyperparams({"working_dir": os.getcwd()})
    trainer = pl.Trainer(
        logger=logger if not TESTING else False,
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        # reload_dataloaders_every_n_epochs=1,
        # callbacks=[SaverCallbackModel(), SaverCallbackEmbnnActionAnchor()],
        callbacks=(
            [
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
                    monitor="val_0/loss",
                    mode="min",
                    save_weights_only=True,
                ),
            ]
            if not TESTING
            else []
        ),
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=5 if "PYTEST_CURRENT_TEST" in os.environ else False,
    )

    dm = MultiviewDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,
        cfg=cfg.dm,
    )

    dm.setup()

    TP_input_dims = Multimodal_ResidualFlow_DiffEmbTransformer.TP_INPUT_DIMS[
        cfg.model.conditioning
    ]
    if cfg.model.conditioning in [
        "latent_z_linear",
        "hybrid_pos_delta_l2norm",
        "hybrid_pos_delta_l2norm_global",
    ]:
        TP_input_dims += (
            cfg.model.latent_z_linear_size
        )  # Hacky way to add the dynamic latent z to the input dims

    # if cfg.conditioning in ["latent_z_linear"]:
    #     assert not cfg.freeze_embnn and not cfg.freeze_z_embnn and not cfg.freeze_residual_flow, "Probably don't want to freeze the network when training the latent model"
    if cfg.model.decoder_type == "taxpose":
        inner_network = ResidualFlow_DiffEmbTransformer(
            emb_dims=cfg.model.emb_dims,
            input_dims=TP_input_dims,
            emb_nn=cfg.model.emb_nn,
            return_flow_component=cfg.model.return_flow_component,
            center_feature=cfg.model.center_feature,
            inital_sampling_ratio=cfg.model.inital_sampling_ratio,
            pred_weight=cfg.model.pred_weight,
            freeze_embnn=cfg.model.freeze_embnn,
            conditioning_size=(
                cfg.model.latent_z_linear_size
                if cfg.model.conditioning
                in [
                    "latent_z_linear_internalcond",
                    "hybrid_pos_delta_l2norm_internalcond",
                    "hybrid_pos_delta_l2norm_global_internalcond",
                ]
                else 0
            ),
            multilaterate=cfg.model.multilaterate,
            sample=cfg.model.mlat_sample,
            mlat_nkps=cfg.model.mlat_nkps,
            pred_mlat_weight=cfg.model.pred_mlat_weight,
            conditioning_type=cfg.model.taxpose_conditioning_type,
            flow_head_use_weighted_sum=cfg.model.flow_head_use_weighted_sum,
            flow_head_use_selected_point_feature=cfg.model.flow_head_use_selected_point_feature,
            post_encoder_input_dims=cfg.model.post_encoder_input_dims,
            flow_direction=cfg.model.flow_direction,
            ghost_points=cfg.model.ghost_points,
            num_ghost_points=cfg.model.num_ghost_points,
            ghost_point_radius=cfg.model.ghost_point_radius,
            relative_3d_encoding=cfg.model.relative_3d_encoding,
        )
    elif cfg.model.decoder_type in ["flow", "point"]:
        inner_network = AlignedFrameDecoder(
            emb_dims=cfg.model.emb_dims,
            input_dims=TP_input_dims,
            flow_direction=cfg.model.flow_direction,
            head_output_type=cfg.model.decoder_type,
            flow_frame=cfg.model.flow_frame,
        )

    network = Multimodal_ResidualFlow_DiffEmbTransformer(
        residualflow_diffembtransformer=inner_network,
        gumbel_temp=cfg.model.gumbel_temp,
        freeze_residual_flow=cfg.model.freeze_residual_flow,
        center_feature=cfg.model.center_feature,
        freeze_z_embnn=cfg.model.freeze_z_embnn,
        division_smooth_factor=cfg.model.division_smooth_factor,
        add_smooth_factor=cfg.model.add_smooth_factor,
        conditioning=cfg.model.conditioning,
        latent_z_linear_size=cfg.model.latent_z_linear_size,
        taxpose_centering=cfg.model.taxpose_centering,
        use_action_z=cfg.model.use_action_z,
        pzY_encoder_type=cfg.model.pzY_encoder_type,
        pzY_dropout_goal_emb=cfg.model.pzY_dropout_goal_emb,
        pzY_transformer=cfg.model.pzY_transformer,
        pzY_transformer_embnn_dims=cfg.model.pzY_transformer_embnn_dims,
        pzY_transformer_emb_dims=cfg.model.pzY_transformer_emb_dims,
        pzY_input_dims=cfg.model.pzY_input_dims,
        pzY_embedding_routine=cfg.model.pzY_embedding_routine,
        pzY_embedding_option=cfg.model.pzY_embedding_option,
        hybrid_cond_logvar_limit=cfg.model.hybrid_cond_logvar_limit,
        latent_z_cond_logvar_limit=cfg.model.latent_z_cond_logvar_limit,
        closest_point_conditioning=cfg.model.pzY_closest_point_conditioning,
    )

    model = EquivarianceTrainingModule(
        network,
        lr=cfg.training.lr,
        image_log_period=cfg.training.image_logging_period,
        flow_supervision=cfg.model.flow_supervision,
        point_loss_type=cfg.model.point_loss_type,
        action_weight=cfg.model.action_weight,
        anchor_weight=cfg.model.anchor_weight,
        displace_weight=cfg.model.displace_weight,
        consistency_weight=cfg.model.consistency_weight,
        smoothness_weight=cfg.model.smoothness_weight,
        rotation_weight=cfg.model.rotation_weight,
        # latent_weight=cfg.model.latent_weight,
        weight_normalize=cfg.model.weight_normalize,
        softmax_temperature=cfg.model.softmax_temperature,
        vae_reg_loss_weight=cfg.model.vae_reg_loss_weight,
        sigmoid_on=cfg.model.sigmoid_on,
        min_err_across_racks_debug=cfg.model.min_err_across_racks_debug,
        error_mode_2rack=cfg.model.error_mode_2rack,
        n_samples=cfg.model.pzY_n_samples,
        get_errors_across_samples=cfg.model.pzY_get_errors_across_samples,
        use_debug_sampling_methods=cfg.model.pzY_use_debug_sampling_methods,
        return_flow_component=cfg.model.return_flow_component,
        plot_encoder_distribution=cfg.model.plot_encoder_distribution,
        joint_infonce_loss_weight=cfg.model.pzY_joint_infonce_loss_weight,
        spatial_distance_regularization_type=cfg.model.spatial_distance_regularization_type,
        spatial_distance_regularization_weight=cfg.model.spatial_distance_regularization_weight,
        hybrid_cond_regularize_all=cfg.model.hybrid_cond_regularize_all,
        pzY_taxpose_infonce_loss_weight=cfg.model.pzY_taxpose_infonce_loss_weight,
        pzY_taxpose_occ_infonce_loss_weight=cfg.model.pzY_taxpose_occ_infonce_loss_weight,
        decoder_type=cfg.model.decoder_type,
        flow_frame=cfg.model.flow_frame,
        compute_rpdiff_min_errors=cfg.model.compute_rpdiff_min_errors,
        rpdiff_descriptions_path=cfg.model.rpdiff_descriptions_path,
    )

    if (
        not cfg.model.pzX_adversarial
        and not cfg.model.joint_train_prior
        and cfg.model.init_cond_x
        and (not cfg.model.freeze_embnn or not cfg.model.freeze_residual_flow)
    ):
        raise ValueError("YOU PROBABLY DIDN'T MEAN TO DO JOINT TRAINING")
    if (
        not cfg.model.joint_train_prior
        and cfg.model.init_cond_x
        and cfg.training.checkpoint_file is None
    ):
        raise ValueError(
            "YOU PROBABLY DIDN'T MEAN TO TRAIN BOTH P(Z|X) AND P(Z|Y) FROM SCRATCH"
        )

    if cfg.model.init_cond_x:
        network_cond_x = Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(
            residualflow_embnn=network,
            encoder_type=cfg.model.pzcondx_encoder_type,
            shuffle_for_pzX=cfg.model.shuffle_for_pzX,
            use_action_z=cfg.model.use_action_z,
            pzX_transformer=cfg.model.pzX_transformer,
            pzX_transformer_embnn_dims=cfg.model.pzX_transformer_embnn_dims,
            pzX_transformer_emb_dims=cfg.model.pzX_transformer_emb_dims,
            pzX_input_dims=cfg.model.pzX_input_dims,
            pzX_dropout_goal_emb=cfg.model.pzX_dropout_goal_emb,
            hybrid_cond_pzX_sample_latent=cfg.model.hybrid_cond_pzX_sample_latent,
            pzX_embedding_routine=cfg.model.pzX_embedding_routine,
            pzX_embedding_option=cfg.model.pzX_embedding_option,
        )

        model_cond_x = EquivarianceTrainingModule_WithPZCondX(
            network_cond_x,
            model,
            goal_emb_cond_x_loss_weight=cfg.model.goal_emb_cond_x_loss_weight,
            joint_train_prior=cfg.model.joint_train_prior,
            freeze_residual_flow=cfg.model.freeze_residual_flow,
            freeze_z_embnn=cfg.model.freeze_z_embnn,
            freeze_embnn=cfg.model.freeze_embnn,
            n_samples=cfg.model.pzX_n_samples,
            get_errors_across_samples=cfg.model.pzX_get_errors_across_samples,
            use_debug_sampling_methods=cfg.model.pzX_use_debug_sampling_methods,
            plot_encoder_distribution=cfg.model.plot_encoder_distribution,
            pzX_use_pzY_z_samples=cfg.model.pzX_use_pzY_z_samples,
            goal_emb_cond_x_loss_type=cfg.model.goal_emb_cond_x_loss_type,
            joint_infonce_loss_weight=cfg.model.pzX_joint_infonce_loss_weight,
            spatial_distance_regularization_type=cfg.model.spatial_distance_regularization_type,
            spatial_distance_regularization_weight=cfg.model.spatial_distance_regularization_weight,
            overwrite_loss=cfg.model.pzX_overwrite_loss,
            pzX_adversarial=cfg.model.pzX_adversarial,
            hybrid_cond_pzX_regularize_type=cfg.model.hybrid_cond_pzX_regularize_type,
            hybrid_cond_pzX_sample_latent=cfg.model.hybrid_cond_pzX_sample_latent,
        )

        model_cond_x.cuda()
        model_cond_x.train()
    else:
        model.cuda()
        model.train()

    if not cfg.training.load_from_checkpoint:
        if cfg.training.checkpoint_file is not None:
            print("loaded checkpoint from")
            print(cfg.training.checkpoint_file)
            if not cfg.model.load_cond_x:
                model.load_state_dict(
                    torch.load(cfg.training.checkpoint_file)["state_dict"]
                )

                # if (
                #     cfg.model.init_cond_x
                #     and cfg.model.load_pretraining_for_conditioning
                # ):
                #     if cfg.training.checkpoint_file_action is not None:
                #         if model_cond_x.model_with_cond_x.encoder_type == "1_dgcnn":
                #             raise NotImplementedError()
                #         elif model_cond_x.model_with_cond_x.encoder_type == "2_dgcnn":
                #             model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.conv5 = nn.Conv2d(
                #                 512, 512, kernel_size=1, bias=False
                #             )
                #             model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.bn5 = nn.BatchNorm2d(
                #                 512
                #             )
                #             model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.load_state_dict(
                #                 torch.load(cfg.training.checkpoint_file_action)[
                #                     "embnn_state_dict"
                #                 ]
                #             )
                #             model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.conv5 = nn.Conv2d(
                #                 512, TP_input_dims - 3, kernel_size=1, bias=False
                #             )
                #             model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.bn5 = nn.BatchNorm2d(
                #                 TP_input_dims - 3
                #             )
                #             print("----Action Pretraining for p(z|X) Loaded!----")
                #         else:
                #             raise ValueError()
                #     if cfg.training.checkpoint_file_anchor is not None:
                #         if model_cond_x.model_with_cond_x.encoder_type == "1_dgcnn":
                #             raise NotImplementedError()
                #         elif model_cond_x.model_with_cond_x.encoder_type == "2_dgcnn":
                #             print("--Not loading p(z|X) for anchor for now--")
                #             pass
                #             # model_cond_x.model_with_cond_x.p_z_cond_x_embnn_anchor.conv5 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
                #             # model_cond_x.model_with_cond_x.p_z_cond_x_embnn_anchor.bn5 = nn.BatchNorm2d(512)
                #             # model_cond_x.model_with_cond_x.p_z_cond_x_embnn_anchor.load_state_dict(
                #             #     torch.load(cfg.training.checkpoint_file_anchor)['embnn_state_dict'])
                #             # model_cond_x.model_with_cond_x.p_z_cond_x_embnn_anchor.conv5 = nn.Conv2d(512, TP_input_dims-3, kernel_size=1, bias=False)
                #             # model_cond_x.model_with_cond_x.p_z_cond_x_embnn_anchor.bn5 = nn.BatchNorm2d(TP_input_dims-3)
                #             # print("--Anchor Pretraining for p(z|X) Loaded!--")
                #         else:
                #             raise ValueError()
            else:
                model_cond_x.load_state_dict(
                    torch.load(cfg.training.checkpoint_file)["state_dict"]
                )

        # else:
        #     if cfg.training.checkpoint_file_action is not None:
        #         if cfg.model.load_pretraining_for_taxpose:
        #             model.model.tax_pose.emb_nn_action.conv1 = nn.Conv2d(
        #                 3 * 2, 64, kernel_size=1, bias=False
        #             )
        #             model.model.tax_pose.emb_nn_action.load_state_dict(
        #                 torch.load(cfg.training.checkpoint_file_action)[
        #                     "embnn_state_dict"
        #                 ]
        #             )
        #             model.model.tax_pose.emb_nn_action.conv1 = nn.Conv2d(
        #                 TP_input_dims * 2, 64, kernel_size=1, bias=False
        #             )
        #             print(
        #                 "-----------------------Pretrained EmbNN Action Model Loaded!-----------------------"
        #             )
        #         if cfg.model.load_pretraining_for_conditioning:
        #             if not cfg.model.init_cond_x:
        #                 print("---Not Loading p(z|Y) Pretraining For Now---")
        #                 pass
        #                 # model.model.emb_nn_objs_at_goal.conv5 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        #                 # model.model.emb_nn_objs_at_goal.bn5 = nn.BatchNorm2d(512)
        #                 # model.model.emb_nn_objs_at_goal.load_state_dict(
        #                 #         torch.load(cfg.training.checkpoint_file_action)['embnn_state_dict'])
        #                 # model.model.emb_nn_objs_at_goal.conv5 = nn.Conv2d(512, TP_input_dims-3, kernel_size=1, bias=False)
        #                 # model.model.emb_nn_objs_at_goal.bn5 = nn.BatchNorm2d(TP_input_dims-3)
        #                 # print("----Action Pretraining for p(z|Y) Loaded!----")
        #             else:
        #                 model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.load_state_dict(
        #                     torch.load(cfg.training.checkpoint_file_action)[
        #                         "embnn_state_dict"
        #                     ]
        #                 )
        #                 print(
        #                     "-----------------------Pretrained p(z|X) Action Encoder Loaded!-----------------------"
        #                 )

        # if cfg.training.checkpoint_file_anchor is not None:
        #     if cfg.model.load_pretraining_for_taxpose:
        #         model.model.tax_pose.emb_nn_anchor.conv1 = nn.Conv2d(
        #             3 * 2, 64, kernel_size=1, bias=False
        #         )
        #         model.model.tax_pose.emb_nn_anchor.load_state_dict(
        #             torch.load(cfg.training.checkpoint_file_anchor)[
        #                 "embnn_state_dict"
        #             ]
        #         )
        #         model.model.tax_pose.emb_nn_anchor.conv1 = nn.Conv2d(
        #             TP_input_dims * 2, 64, kernel_size=1, bias=False
        #         )
        #         print(
        #             "-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------"
        #         )
        #     if cfg.model.load_pretraining_for_conditioning:
        #         if not cfg.model.init_cond_x:
        #             print("---Not Loading p(z|Y) Pretraining For Now---")
        #             pass
        #             # if cfg.training.checkpoint_file_action is None:
        #             #     model.model.emb_nn_objs_at_goal.conv5 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        #             #     model.model.emb_nn_objs_at_goal.bn5 = nn.BatchNorm2d(512)
        #             #     model.model.emb_nn_objs_at_goal.load_state_dict(
        #             #             torch.load(cfg.training.checkpoint_file_action)['embnn_state_dict'])
        #             #     model.model.emb_nn_objs_at_goal.conv5 = nn.Conv2d(512, TP_input_dims-3, kernel_size=1, bias=False)
        #             #     model.model.emb_nn_objs_at_goal.bn5 = nn.BatchNorm2d(TP_input_dims-3)
        #             #     print("----Anchor Pretraining for p(z|Y) Loaded! (because action pretraining is not present)----")
        #         else:
        #             model_cond_x.model_with_cond_x.p_z_cond_x_embnn_anchor.load_state_dict(
        #                 torch.load(cfg.training.checkpoint_file_anchor)[
        #                     "embnn_state_dict"
        #                 ]
        #             )
        #             print(
        #                 "-----------------------Pretrained p(z|X) Anchor Encoder Loaded!-----------------------"
        #             )

    if cfg.model.init_cond_x:
        return trainer, model_cond_x, dm
    else:
        return trainer, model, dm


@hydra.main(version_base="1.1", config_path="../configs", config_name="train_ndf")
def main(cfg):
    trainer, model, dm = setup_main(cfg)

    resume_training_ckpt = (
        cfg.training.checkpoint_file if cfg.training.load_from_checkpoint else None
    )

    restarts = 0
    while restarts < 1:
        trainer.fit(model, dm, ckpt_path=resume_training_ckpt)

        if True or trainer.current_epoch > trainer.max_epochs:
            # This doesn't need a restart. it finished gracefully
            return

        print(
            f"\nTrainer finished. Restarting because current epoch {trainer.current_epoch} is less than max epochs {trainer.max_epochs}"
        )

        # Get the latest checkpoint
        folder = os.path.join(
            trainer.logger.experiment.dir, "..", "..", "..", "residual_flow_occlusion"
        )
        if not os.path.isdir(folder):
            print(f"\nDidn't find the checkpoint folder in {folder}. Quitting.")
            return
        subfolder = os.listdir(folder)[0]
        folder = os.path.join(folder, subfolder, "checkpoints")
        checkpoints = [f for f in os.listdir(folder) if f.startswith("epoch=")]
        if len(checkpoints) == 0:
            print(
                f"\nDidn't find a checkpoint file in {folder}. Assuming the script crashed before the first save. Quitting."
            )
            return

        checkpoint = checkpoints[0]

        del trainer
        del model
        del dm
        import gc

        gc.collect()

        checkpoint_file = os.path.join(folder, checkpoint)
        print("\nRestarting script from latest checkpoint:", checkpoint_file)
        cfg.checkpoint_file = checkpoint_file
        trainer, model, dm = setup_main(cfg)

    print(f"\nAlready restarted {restarts} times. Quitting")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
