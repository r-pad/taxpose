class ResidualFlow_DiffEmbTransformer(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        cycle=True,
        emb_nn="dgcnn",
        return_flow_component=False,
        center_feature=False,
        pred_weight=True,
        residual_on=True,
        freeze_embnn=False,
        return_attn=True,
        multilaterate=False,
        sample: bool = False,
        mlat_nkps: int = 100,
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.cycle = cycle
        if emb_nn == "dgcnn":
            self.emb_nn_action = DGCNN(emb_dims=self.emb_dims)
            self.emb_nn_anchor = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not implemented")
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn
        self.return_attn = return_attn

        self.transformer_action = CustomTransformer(
            emb_dims=emb_dims, return_attn=self.return_attn, bidirectional=False
        )
        self.transformer_anchor = CustomTransformer(
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
