import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.taxposed_dgcnn import DGCNN, DGCNNClassification


class Multimodal_ResidualFlow_DiffEmbTransformer(nn.Module):
    EMB_DIMS_BY_CONDITIONING = {
        "pos_delta_l2norm": 1,
        "uniform_prior_pos_delta_l2norm": 1,
        # 'latent_z': 1, # Make the dimensions as close as possible to the ablations we're comparing this against
        # 'latent_z_1pred': 1, # Same
        # 'latent_z_1pred_10d': 10, # Same
        "latent_z_linear": 512,
        "latent_z_linear_internalcond": 512,
        "pos_delta_vec": 1,
        "pos_onehot": 1,
        "pos_loc3d": 3,
    }

    # Number of heads that the DGCNN should output
    NUM_HEADS_BY_CONDITIONING = {
        "pos_delta_l2norm": 1,
        "uniform_prior_pos_delta_l2norm": 1,
        # 'latent_z': 2, # One for mu and one for var
        # 'latent_z_1pred': 2, # Same
        # 'latent_z_1pred_10d': 2, # Same
        "latent_z_linear": 2,
        "latent_z_linear_internalcond": 2,
        "pos_delta_vec": 1,
        "pos_onehot": 1,
        "pos_loc3d": 1,
    }

    DEPRECATED_CONDITIONINGS = ["latent_z", "latent_z_1pred", "latent_z_1pred_10d"]

    TP_INPUT_DIMS = {
        "pos_delta_l2norm": 3 + 1,
        "uniform_prior_pos_delta_l2norm": 3 + 1,
        # Not implemented because it's dynamic. Also this isn't used anymore
        # 'latent_z_linear': 3 + cfg.latent_z_linear_size,
        "latent_z_linear_internalcond": 3,
        "pos_delta_vec": 3 + 3,
        "pos_onehot": 3 + 1,
        "pos_loc3d": 3 + 3,
        "latent_3d_z": 3 + 3,
    }

    def __init__(
        self,
        residualflow_diffembtransformer,
        gumbel_temp=0.5,
        freeze_residual_flow=False,
        center_feature=False,
        freeze_z_embnn=False,
        division_smooth_factor=1,
        add_smooth_factor=0.05,
        conditioning="pos_delta_l2norm",
        latent_z_linear_size=40,
        taxpose_centering="mean",
    ):
        super(Multimodal_ResidualFlow_DiffEmbTransformer, self).__init__()

        assert taxpose_centering in ["mean", "z"]
        assert (
            conditioning not in self.DEPRECATED_CONDITIONINGS
        ), f"This conditioning {conditioning} is deprecated and should not be used"
        assert conditioning in self.EMB_DIMS_BY_CONDITIONING.keys()

        self.latent_z_linear_size = latent_z_linear_size
        self.conditioning = conditioning
        self.taxpose_centering = taxpose_centering
        # if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
        #     assert not freeze_residual_flow and not freeze_z_embnn, "Prob didn't want to freeze residual flow or z embnn when using latent_z_linear"

        self.tax_pose = residualflow_diffembtransformer

        self.emb_dims = self.EMB_DIMS_BY_CONDITIONING[self.conditioning]
        self.num_emb_heads = self.NUM_HEADS_BY_CONDITIONING[self.conditioning]
        # Point cloud with class labels between action and anchor
        if self.conditioning not in ["latent_z_linear", "latent_z_linear_internalcond"]:
            self.emb_nn_objs_at_goal = DGCNN(
                emb_dims=self.emb_dims, num_heads=self.num_emb_heads, last_relu=False
            )
        else:
            self.emb_nn_objs_at_goal = DGCNNClassification(
                emb_dims=self.emb_dims,
                num_heads=self.num_emb_heads,
                dropout=0.5,
                output_channels=self.latent_z_linear_size,
            )
        # TODO
        self.freeze_residual_flow = freeze_residual_flow
        self.center_feature = center_feature
        self.freeze_z_embnn = freeze_z_embnn
        self.freeze_embnn = self.tax_pose.freeze_embnn
        self.gumbel_temp = gumbel_temp

        self.division_smooth_factor = division_smooth_factor
        self.add_smooth_factor = add_smooth_factor

    def get_dense_translation_point(self, points, ref, conditioning):
        """
        points- point cloud. (B, 3, num_points)
        ref- one hot vector (or nearly one-hot) that denotes the reference point
                 (B, num_points)

        Returns:
            dense point cloud. Each point contains the distance to the reference point (B, 3 or 1, num_points)
        """
        assert ref.ndim == 2
        assert torch.allclose(
            ref.sum(axis=1),
            torch.full((ref.shape[0], 1), 1, dtype=torch.float, device=ref.device),
        )
        num_points = points.shape[2]
        reference = (points * ref[:, None, :]).sum(axis=2)
        if conditioning in ["pos_delta_l2norm", "uniform_prior_pos_delta_l2norm"]:
            dense = torch.norm(reference[:, :, None] - points, dim=1, keepdim=True)
        elif conditioning == "pos_delta_vec":
            dense = reference[:, :, None] - points
        elif conditioning == "pos_loc3d":
            dense = reference[:, :, None].repeat(1, 1, 1024)
        elif conditioning == "pos_onehot":
            dense = ref[:, None, :]
        else:
            raise ValueError(
                f"Conditioning {conditioning} probably doesn't require a dense representation. This function is for"
                + "['pos_delta_l2norm', 'pos_delta_vec', 'pos_loc3d', 'pos_onehot', 'uniform_prior_pos_delta_l2norm']"
            )
        return dense, reference

    def add_conditioning(self, goal_emb, action_points, anchor_points, conditioning):
        for_debug = {}

        if conditioning in [
            "pos_delta_l2norm",
            "pos_delta_vec",
            "pos_loc3d",
            "pos_onehot",
            "uniform_prior_pos_delta_l2norm",
        ]:
            goal_emb = (goal_emb + self.add_smooth_factor) / self.division_smooth_factor

            # Only handle the translation case for now
            goal_emb_translation = goal_emb[:, 0, :]

            goal_emb_translation_action = goal_emb_translation[
                :, : action_points.shape[2]
            ]
            goal_emb_translation_anchor = goal_emb_translation[
                :, action_points.shape[2] :
            ]

            translation_sample_action = F.gumbel_softmax(
                goal_emb_translation_action, self.gumbel_temp, hard=True, dim=-1
            )
            translation_sample_anchor = F.gumbel_softmax(
                goal_emb_translation_anchor, self.gumbel_temp, hard=True, dim=-1
            )

            # This is the only line that's different among the 3 different conditioning schemes in this category
            (
                dense_trans_pt_action,
                ref_action,
            ) = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                None,
                action_points,
                translation_sample_action,
                conditioning=self.conditioning,
            )
            (
                dense_trans_pt_anchor,
                ref_anchor,
            ) = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                None,
                anchor_points,
                translation_sample_anchor,
                conditioning=self.conditioning,
            )

            action_points_and_cond = torch.cat(
                [action_points] + [dense_trans_pt_action], axis=1
            )
            anchor_points_and_cond = torch.cat(
                [anchor_points] + [dense_trans_pt_anchor], axis=1
            )

            for_debug = {
                "dense_trans_pt_action": dense_trans_pt_action,
                "dense_trans_pt_anchor": dense_trans_pt_anchor,
                "trans_pt_action": ref_action,
                "trans_pt_anchor": ref_anchor,
                "trans_sample_action": translation_sample_action,
                "trans_sample_anchor": translation_sample_anchor,
            }
        elif conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            # Do the reparametrization trick on the predicted mu and var

            # Here, the goal emb has 2 heads. One for mean and one for variance
            goal_emb_mu = goal_emb[0]
            goal_emb_logvar = goal_emb[1]

            def reparametrize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps * std + mu

            goal_emb = reparametrize(goal_emb_mu, goal_emb_logvar)

            for_debug = {
                "goal_emb_mu": goal_emb_mu,
                "goal_emb_logvar": goal_emb_logvar,
            }

            if conditioning == "latent_z_linear":
                action_points_and_cond = torch.cat(
                    [action_points]
                    + [torch.tile(goal_emb, (1, 1, action_points.shape[-1]))],
                    axis=1,
                )
                anchor_points_and_cond = torch.cat(
                    [anchor_points]
                    + [torch.tile(goal_emb, (1, 1, anchor_points.shape[-1]))],
                    axis=1,
                )
            elif conditioning == "latent_z_linear_internalcond":
                # The cond will be added in by TAXPose
                action_points_and_cond = action_points
                anchor_points_and_cond = anchor_points
                for_debug["goal_emb"] = goal_emb
            else:
                raise ValueError("Why is it here?")
        else:
            raise ValueError(
                f"Conditioning {conditioning} does not exist. Choose one of: {list(self.EMB_DIMS_BY_CONDITIONING.keys())}"
            )

        return action_points_and_cond, anchor_points_and_cond, for_debug

    def forward(self, *input, mode="forward"):
        # Forward pass goes through all of the model
        # Inference will use a sample from the prior if there is one
        #     - ex: conditioning = latent_z_linear_internalcond
        assert mode in ["forward", "inference"]

        action_points = input[0].permute(0, 2, 1)[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]

        if input[2] is None:
            mode = "inference"

        if mode == "forward":
            goal_action_points = input[2].permute(0, 2, 1)[:, :3]
            goal_anchor_points = input[3].permute(0, 2, 1)[:, :3]

            # mean center point cloud before DGCNN
            if self.center_feature:
                mean_goal = torch.cat(
                    [goal_action_points, goal_anchor_points], axis=-1
                ).mean(dim=2, keepdim=True)
                goal_action_points_dmean = goal_action_points - mean_goal
                goal_anchor_points_dmean = goal_anchor_points - mean_goal
                action_points_dmean = action_points - action_points.mean(
                    dim=2, keepdim=True
                )
                anchor_points_dmean = anchor_points - anchor_points.mean(
                    dim=2, keepdim=True
                )
            else:
                goal_action_points_dmean = goal_action_points
                goal_anchor_points_dmean = goal_anchor_points
                action_points_dmean = action_points
                anchor_points_dmean = anchor_points

            goal_points_dmean = torch.cat(
                [goal_action_points_dmean, goal_anchor_points_dmean], axis=2
            )

            if self.freeze_z_embnn:
                with torch.no_grad():
                    if self.num_emb_heads > 1:
                        goal_emb = [
                            a.detach()
                            for a in self.emb_nn_objs_at_goal(goal_points_dmean)
                        ]
                    else:
                        goal_emb = self.emb_nn_objs_at_goal(goal_points_dmean).detach()
            else:
                goal_emb = self.emb_nn_objs_at_goal(goal_points_dmean)

            (
                action_points_and_cond,
                anchor_points_and_cond,
                for_debug,
            ) = self.add_conditioning(
                goal_emb, action_points, anchor_points, self.conditioning
            )
        elif mode == "inference":
            (
                action_points_and_cond,
                anchor_points_and_cond,
                goal_emb,
                for_debug,
            ) = self.sample(action_points, anchor_points)
        else:
            raise ValueError(f"Unknown mode {mode}")

        tax_pose_conditioning_action = None
        tax_pose_conditioning_anchor = None
        if self.conditioning == "latent_z_linear_internalcond":
            tax_pose_conditioning_action = torch.tile(
                for_debug["goal_emb"], (1, 1, action_points.shape[-1])
            )
            tax_pose_conditioning_anchor = torch.tile(
                for_debug["goal_emb"], (1, 1, anchor_points.shape[-1])
            )

        if self.taxpose_centering == "mean":
            # Use TAX-Pose defaults
            action_center = action_points[:, :3].mean(dim=2, keepdim=True)
            anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)
        elif self.taxpose_centering == "z":
            action_center = for_debug["trans_pt_action"][:, :, None]
            anchor_center = for_debug["trans_pt_anchor"][:, :, None]
        else:
            raise ValueError(
                f"Unknown self.taxpose_centering: {self.taxpose_centering}"
            )

        if self.freeze_residual_flow:
            with torch.no_grad():
                flow_action = self.tax_pose(
                    action_points_and_cond.permute(0, 2, 1),
                    anchor_points_and_cond.permute(0, 2, 1),
                    conditioning_action=tax_pose_conditioning_action,
                    conditioning_anchor=tax_pose_conditioning_anchor,
                    action_center=action_center,
                    anchor_center=anchor_center,
                )
        else:
            flow_action = self.tax_pose(
                action_points_and_cond.permute(0, 2, 1),
                anchor_points_and_cond.permute(0, 2, 1),
                conditioning_action=tax_pose_conditioning_action,
                conditioning_anchor=tax_pose_conditioning_anchor,
                action_center=action_center,
                anchor_center=anchor_center,
            )

        ########## LOGGING ############

        # Change goal_emb here to be what is going to be logged. For the latent_z conditioning, we just log the mean
        if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            goal_emb = goal_emb[0]

        if self.freeze_residual_flow:
            flow_action["flow_action"] = flow_action["flow_action"].detach()
            flow_action["flow_anchor"] = flow_action["flow_anchor"].detach()

        flow_action = {
            **flow_action,
            "goal_emb": goal_emb,
            **for_debug,
        }
        return flow_action

    def sample(self, action_points, anchor_points):
        if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            # Take a SINGLE sample z ~ N(0,1)
            for_debug = {}
            goal_emb_action = None
            goal_emb_anchor = None
            if self.conditioning == "latent_z_linear":
                goal_emb = torch.tile(
                    torch.randn((action_points.shape[0], self.emb_dims, 1)).to(
                        action_points.device
                    ),
                    (1, 1, action_points.shape[-1]),
                )
                action_points_and_cond = torch.cat([action_points, goal_emb], axis=1)
                anchor_points_and_cond = torch.cat([anchor_points, goal_emb], axis=1)
            elif self.conditioning == "latent_z_linear_internalcond":
                goal_emb = torch.randn(
                    (action_points.shape[0], self.latent_z_linear_size, 1)
                ).to(action_points.device)
                action_points_and_cond = action_points
                anchor_points_and_cond = anchor_points
                for_debug["goal_emb"] = goal_emb
            else:
                raise ValueError("Why is it here?")
        elif self.conditioning in ["uniform_prior_pos_delta_l2norm"]:
            # sample from a uniform prior
            N_action, N_anchor, B = (
                action_points.shape[-1],
                anchor_points.shape[-1],
                action_points.shape[0],
            )
            translation_sample_action = (
                F.one_hot(torch.randint(N_action, (B,)), N_action).float().cuda()
            )
            translation_sample_anchor = (
                F.one_hot(torch.randint(N_anchor, (B,)), N_anchor).float().cuda()
            )

            (
                dense_trans_pt_action,
                ref_action,
            ) = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                None,
                action_points,
                translation_sample_action,
                conditioning=self.conditioning,
            )
            (
                dense_trans_pt_anchor,
                ref_anchor,
            ) = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                None,
                anchor_points,
                translation_sample_anchor,
                conditioning=self.conditioning,
            )

            action_points_and_cond = torch.cat(
                [action_points] + [dense_trans_pt_action], axis=1
            )
            anchor_points_and_cond = torch.cat(
                [anchor_points] + [dense_trans_pt_anchor], axis=1
            )

            goal_emb = None

            for_debug = {
                "dense_trans_pt_action": dense_trans_pt_action,
                "dense_trans_pt_anchor": dense_trans_pt_anchor,
                "trans_pt_action": ref_action,
                "trans_pt_anchor": ref_anchor,
                "trans_sample_action": translation_sample_action,
                "trans_sample_anchor": translation_sample_anchor,
            }
        else:
            raise ValueError(
                f"Sampling not supported for conditioning {self.conditioning}. Pick one of the latent_z_xxx conditionings"
            )
        return action_points_and_cond, anchor_points_and_cond, goal_emb, for_debug


class Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(nn.Module):
    def __init__(
        self,
        residualflow_embnn,
        encoder_type="2_dgcnn",
        sample_z=True,
        shuffle_for_pzX=False,
        return_debug=False,
    ):
        super(Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX, self).__init__()
        self.residflow_embnn = residualflow_embnn

        # Use the other class definition so that it matches between classes
        self.conditioning = self.residflow_embnn.conditioning
        self.num_emb_heads = self.residflow_embnn.num_emb_heads
        self.emb_dims = self.residflow_embnn.emb_dims
        self.taxpose_centering = self.residflow_embnn.taxpose_centering
        self.freeze_residual_flow = self.residflow_embnn.freeze_residual_flow
        self.freeze_z_embnn = self.residflow_embnn.freeze_z_embnn
        self.freeze_embnn = self.residflow_embnn.freeze_embnn

        self.shuffle_for_pzX = shuffle_for_pzX
        self.return_debug = return_debug

        # assert self.conditioning not in ['uniform_prior_pos_delta_l2norm']

        # assert self.conditioning not in ["latent_z_linear", "latent_z", "latent_z_1pred", "latent_z_1pred_10d", "latent_z_linear_internalcond"], "Latent z conditioning does not need a p(z|X) because it's regularized to N(0,1)"

        # Note: 1 DGCNN probably loses some of the rotational invariance between objects
        assert encoder_type in ["1_dgcnn", "2_dgcnn"]

        # disable smoothing
        self.add_smooth_factor = 0.05
        self.division_smooth_factor = 1.0
        self.gumbel_temp = self.residflow_embnn.gumbel_temp

        self.encoder_type = encoder_type
        self.sample_z = sample_z

        if self.conditioning not in ["latent_z_linear", "latent_z_linear_internalcond"]:
            if self.encoder_type == "1_dgcnn":
                self.p_z_cond_x_embnn = DGCNN(
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
            elif self.encoder_type == "2_dgcnn":
                self.p_z_cond_x_embnn_action = DGCNN(
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
                self.p_z_cond_x_embnn_anchor = DGCNN(
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    last_relu=False,
                )
            else:
                raise ValueError()
        else:
            if self.encoder_type == "1_dgcnn":
                self.p_z_cond_x_embnn = DGCNNClassification(
                    emb_dims=self.emb_dims, num_heads=self.num_emb_heads
                )
            elif self.encoder_type == "2_dgcnn":
                self.p_z_cond_x_embnn_action = DGCNNClassification(
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    dropout=0.5,
                    output_channels=self.residflow_embnn.latent_z_linear_size,
                )
                self.p_z_cond_x_embnn_anchor = DGCNNClassification(
                    emb_dims=self.emb_dims,
                    num_heads=self.num_emb_heads,
                    dropout=0.5,
                    output_channels=self.residflow_embnn.latent_z_linear_size,
                )
            else:
                raise ValueError()

        self.center_feature = self.residflow_embnn.center_feature

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]

        # mean center point cloud before DGCNN
        if self.residflow_embnn.center_feature:
            action_points_dmean = action_points - action_points.mean(
                dim=2, keepdim=True
            )
            anchor_points_dmean = anchor_points - anchor_points.mean(
                dim=2, keepdim=True
            )
        else:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points

        if self.shuffle_for_pzX:
            action_shuffle_idxs = torch.randperm(action_points_dmean.size()[2])
            anchor_shuffle_idxs = torch.randperm(anchor_points_dmean.size()[2])
            action_points_dmean = action_points_dmean[:, :, action_shuffle_idxs]
            anchor_points_dmean = anchor_points_dmean[:, :, anchor_shuffle_idxs]

        def prepare(arr, is_action):
            if self.shuffle_for_pzX:
                shuffle_idxs = action_shuffle_idxs if is_action else anchor_shuffle_idxs
                return arr[:, :, torch.argsort(shuffle_idxs)]
            else:
                return arr

        if self.encoder_type == "1_dgcnn":
            goal_emb_cond_x = self.p_z_cond_x_embnn(
                torch.cat([action_points_dmean, anchor_points_dmean], dim=-1)
            )
            goal_emb_cond_x_action = prepare(
                goal_emb_cond_x[:, :, : action_points_dmean.shape[-1]]
            )
            goal_emb_cond_x_anchor = prepare(
                goal_emb_cond_x[:, :, action_points_dmean.shape[-1] :]
            )
        elif self.encoder_type == "2_dgcnn":
            # Sample a point
            goal_emb_cond_x_action = self.p_z_cond_x_embnn_action(action_points_dmean)
            goal_emb_cond_x_anchor = self.p_z_cond_x_embnn_anchor(anchor_points_dmean)

            if self.num_emb_heads > 1:
                goal_emb_cond_x = [
                    torch.cat(
                        [prepare(action_head, True), prepare(anchor_head, False)],
                        dim=-1,
                    )
                    for action_head, anchor_head in zip(
                        goal_emb_cond_x_action, goal_emb_cond_x_anchor
                    )
                ]
            else:
                goal_emb_cond_x = torch.cat(
                    [
                        prepare(goal_emb_cond_x_action, True),
                        prepare(goal_emb_cond_x_anchor, False),
                    ],
                    dim=-1,
                )
        else:
            raise ValueError()

        (
            action_points_and_cond,
            anchor_points_and_cond,
            for_debug,
        ) = Multimodal_ResidualFlow_DiffEmbTransformer.add_conditioning(
            self, goal_emb_cond_x, action_points, anchor_points, self.conditioning
        )

        tax_pose_conditioning_action = None
        tax_pose_conditioning_anchor = None
        if self.conditioning == "latent_z_linear_internalcond":
            tax_pose_conditioning_action = torch.tile(
                for_debug["goal_emb"][:, :, 0][:, :, None],
                (1, 1, action_points.shape[-1]),
            )
            tax_pose_conditioning_anchor = torch.tile(
                for_debug["goal_emb"][:, :, 1][:, :, None],
                (1, 1, anchor_points.shape[-1]),
            )

        if self.taxpose_centering == "mean":
            # Use TAX-Pose defaults
            action_center = action_points[:, :3].mean(dim=2, keepdim=True)
            anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)
        elif self.taxpose_centering == "z":
            action_center = for_debug["trans_pt_action"][:, :, None]
            anchor_center = for_debug["trans_pt_anchor"][:, :, None]
        else:
            raise ValueError(
                f"Unknown self.taxpose_centering: {self.taxpose_centering}"
            )

        # Unpermute the action and anchor point clouds to match how tax pose is written
        flow_action = self.residflow_embnn.tax_pose(
            action_points_and_cond.permute(0, 2, 1),
            anchor_points_and_cond.permute(0, 2, 1),
            conditioning_action=tax_pose_conditioning_action,
            conditioning_anchor=tax_pose_conditioning_anchor,
            action_center=action_center,
            anchor_center=anchor_center,
        )

        # If the demo is available, run p(z|Y)
        if input[2] is not None:
            # Inputs 2 and 3 are the objects in demo positions
            # If we have access to these, we can run the pzY network
            pzY_results = self.residflow_embnn(*input)
            goal_emb = pzY_results["goal_emb"]
        else:
            goal_emb = None

        flow_action = {
            **flow_action,
            "goal_emb": goal_emb,
            "goal_emb_cond_x": goal_emb_cond_x,
            **for_debug,
        }
        return flow_action
