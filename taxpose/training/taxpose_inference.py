from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch

from taxpose.datasets.ndf import compute_demo_symmetry_features
from taxpose.models.taxpose_reasoning import SymmetryConfig, TAXPoseReasoning


class TAXPoseInferenceModule(pl.LightningModule):
    def __init__(
        self,
        reasoning_module: TAXPoseReasoning,
        symmetry_cfg: Optional[SymmetryConfig] = None,
    ) -> None:
        super().__init__()

        self.model = reasoning_module

        self.symmetry_cfg = symmetry_cfg

    def forward(
        self,
        points_action: torch.Tensor,
        points_anchor: torch.Tensor,
        features_action: Optional[torch.Tensor] = None,
        features_anchor: Optional[torch.Tensor] = None,
    ) -> Dict:
        if self.symmetry_cfg and (features_action is None or features_anchor is None):
            raise ValueError("NEVER compute symmetry features before downsampling...")
            points_action_np = points_action.cpu().numpy()
            points_anchor_np = points_anchor.cpu().numpy()

            # Iterate
            features_action_list = []
            features_anchor_list = []
            for i in range(points_action_np.shape[0]):
                # TODO: Incorporate color somehow...
                (
                    features_action,
                    features_anchor,
                    action_symmetry_rgb,
                    anchor_symmetry_rgb,
                ) = compute_demo_symmetry_features(
                    points_action_np[i : i + 1],
                    points_anchor_np[i : i + 1],
                    self.symmetry_cfg.object_type,
                    self.symmetry_cfg.action,
                    self.symmetry_cfg.action_class,
                    self.symmetry_cfg.anchor_class,
                    self.symmetry_cfg.normalize_dist,
                )
                features_action_list.append(features_action)
                features_anchor_list.append(features_anchor)

            features_action = np.concatenate(features_action_list, axis=0)
            features_anchor = np.concatenate(features_anchor_list, axis=0)

            features_action = torch.from_numpy(features_action).to(points_action.device)
            features_anchor = torch.from_numpy(features_anchor).to(points_anchor.device)

        return self.model(
            points_action,
            points_anchor,
            features_action,
            features_anchor,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        points_action = batch["points_action"]
        points_anchor = batch["points_anchor"]
        features_action = batch["features_action"]
        features_anchor = batch["features_anchor"]

        return self(points_action, points_anchor, features_action, features_anchor)
