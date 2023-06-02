class ResidualFlow_DiffEmbTransformer(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        return_flow_component=False,
        center_feature=False,
        inital_sampling_ratio=0.2,
        pred_weight=True,
        residual_on=True,
        freeze_embnn=False,
        return_attn=True,
        input_dims=3,
        multilaterate=False,
        sample: bool = False,
        mlat_nkps: int = 100,
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        self.input_dims = input_dims
        if emb_nn == "pointnet":
            self.emb_nn_action = PointNet()
            self.emb_nn_anchor = PointNet()
        elif emb_nn == "pointnet2ben":
            params_action = PN2DenseParams()
            params_action.sa1.ratio = inital_sampling_ratio
            self.emb_nn_action = Pointnet2Dense(emb_dims, params=params_action)
            params_anchor = PN2DenseParams()
            params_anchor.sa1.ratio = inital_sampling_ratio
            self.emb_nn_anchor = Pointnet2Dense(emb_dims, params=params_anchor)
        elif emb_nn == "pointnet2ori":
            self.emb_nn_action = pointnet2(emb_dims)
            self.emb_nn_anchor = pointnet2(emb_dims)
        elif emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(
                emb_dims=self.emb_dims, input_dims=self.input_dims
            )
            self.emb_nn_anchor = DGCNN(
                emb_dims=self.emb_dims, input_dims=self.input_dims
            )
        elif emb_nn == "vn":
            self.emb_nn_action = VN_PointNet()
            self.emb_nn_anchor = VN_PointNet()
        else:
            raise Exception("Not implemented")
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn
        self.return_attn = return_attn

        self.transformer_action = Transformer(
            emb_dims=emb_dims, return_attn=self.return_attn, bidirectional=False
        )
        self.transformer_anchor = Transformer(
            emb_dims=emb_dims, return_attn=self.return_attn, bidirectional=False
        )
        if multilaterate:
            self.head_action = MultilaterationHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                sample=sample,
                n_kps=mlat_nkps,
            )
            self.head_anchor = MultilaterationHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                sample=sample,
                n_kps=mlat_nkps,
            )
        else:
            self.head_action = ResidualMLPHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                residual_on=self.residual_on,
            )
            self.head_anchor = ResidualMLPHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                residual_on=self.residual_on,
            )

    def forward(self, *input):
        assert action_points.shape[1] in [
            3,
            4,
        ], "action_points should be of shape (Batch, {3,4}. num_points), but got {}".format(
            action_points.shape
        )
        # compute action_dmean
        if action_points.shape[1] == 4:
            raise ValueError("this is not expected...")
            # action_xyz = action_points_input[:, :3]
            # anchor_xyz = anchor_points_input[:, :3]
            # action_sym_cls = action_points_input[:, 3:]
            # anchor_sym_cls = anchor_points_input[:, 3:]
            # action_xyz_dmean = action_xyz - action_xyz.mean(dim=2, keepdim=True)
            # anchor_xyz_dmean = anchor_xyz - anchor_xyz.mean(dim=2, keepdim=True)
            # action_points_dmean = torch.cat([action_xyz_dmean, action_sym_cls], axis=1)
            # anchor_points_dmean = torch.cat([anchor_xyz_dmean, anchor_sym_cls], axis=1)

        # elif action_points.shape[1] == 3:

        action_points = input[0].permute(0, 2, 1)[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]

        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)

        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points

        action_embedding = self.emb_nn_action(action_points_dmean)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)

        if self.freeze_embnn:
            action_embedding = action_embedding.detach()
            anchor_embedding = anchor_embedding.detach()

        # tilde_phi, phi are both B,512,N
        # Get the new cross-attention embeddings.
        transformer_action_outputs = self.transformer_action(
            action_embedding, anchor_embedding
        )
        transformer_anchor_outputs = self.transformer_anchor(
            anchor_embedding, action_embedding
        )
        action_embedding_tf = transformer_action_outputs["src_embedding"]
        action_attn = transformer_action_outputs["src_attn"]
        anchor_embedding_tf = transformer_anchor_outputs["src_embedding"]
        anchor_attn = transformer_anchor_outputs["src_attn"]

        if not self.return_attn:
            action_attn = None
            anchor_attn = None

        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        if action_attn is not None:
            action_attn = action_attn.mean(dim=1)

        head_action_output = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=action_attn,
        )
        flow_action = head_action_output["full_flow"].permute(0, 2, 1)
        residual_flow_action = head_action_output["residual_flow"].permute(0, 2, 1)
        corr_flow_action = head_action_output["corr_flow"].permute(0, 2, 1)
        corr_points_action = head_action_output["corr_points"].permute(0, 2, 1)

        outputs = {
            "flow_action": flow_action,
            "residual_flow_action": residual_flow_action,
            "corr_flow_action": corr_flow_action,
            "corr_points_action": corr_points_action,
        }

        if "P_A" in head_action_output:
            original_points_action = head_action_output["P_A"].permute(0, 2, 1)
            outputs["original_points_action"] = original_points_action
            outputs["sampled_ixs_action"] = head_action_output["A_ixs"]

        if self.cycle:
            if anchor_attn is not None:
                anchor_attn = anchor_attn.mean(dim=1)

            head_anchor_output = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=anchor_attn,
            )
            flow_anchor = head_anchor_output["full_flow"].permute(0, 2, 1)
            residual_flow_anchor = head_anchor_output["residual_flow"].permute(0, 2, 1)
            corr_flow_anchor = head_anchor_output["corr_flow"].permute(0, 2, 1)
            corr_points_anchor = head_anchor_output["corr_points"].permute(0, 2, 1)

            outputs = {
                **outputs,
                "flow_anchor": flow_anchor,
                "residual_flow_anchor": residual_flow_anchor,
                "corr_flow_anchor": corr_flow_anchor,
                "corr_points_anchor": corr_points_anchor,
            }

            if "P_A" in head_anchor_output:
                original_points_anchor = head_anchor_output["P_A"].permute(0, 2, 1)
                outputs["original_points_anchor"] = original_points_anchor
                outputs["sampled_ixs_anchor"] = head_anchor_output["A_ixs"]

        return outputs
