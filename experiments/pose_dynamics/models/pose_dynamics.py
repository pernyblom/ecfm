from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from .cnn import SmallCNN


def _rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    yaw = rpy[:, 2]

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


class PoseDynamicsProjector(nn.Module):
    def __init__(
        self,
        reps: List[str],
        cnn_channels: List[int],
        feature_dim: int,
        hidden_dim: int,
        history_steps: int,
        future_steps: int,
        min_depth: float = 0.1,
    ) -> None:
        super().__init__()
        self.reps = reps
        self.history_steps = int(history_steps)
        self.future_steps = int(future_steps)
        self.min_depth = float(min_depth)

        self.encoders = nn.ModuleDict(
            {rep: SmallCNN(in_channels=3, channels=cnn_channels, feature_dim=feature_dim) for rep in reps}
        )

        history_points = self.history_steps + 1
        history_input_dim = history_points * 2 + max(self.history_steps, 0) * 2
        self.history_encoder = nn.Sequential(
            nn.Linear(history_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = feature_dim * (len(reps) + 1)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.pose_delta_head = nn.Linear(hidden_dim, 6)
        self.intr_delta_head = nn.Linear(hidden_dim, 4)
        self.pos_head = nn.Linear(hidden_dim, 3)
        self.vel_head = nn.Linear(hidden_dim, 3)
        self.acc_head = nn.Linear(hidden_dim, self.future_steps * 3)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        past_centers: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_pose: torch.Tensor,
        dt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats = []
        for rep in self.reps:
            feats.append(self.encoders[rep](inputs[rep]))

        history_flat = past_centers.reshape(past_centers.shape[0], -1)
        if past_centers.shape[1] > 1:
            history_delta = past_centers[:, 1:] - past_centers[:, :-1]
            history_delta_flat = history_delta.reshape(history_delta.shape[0], -1)
        else:
            history_delta_flat = past_centers.new_zeros((past_centers.shape[0], 0))
        history_feat = self.history_encoder(torch.cat([history_flat, history_delta_flat], dim=-1))

        fused = self.fusion(torch.cat([*feats, history_feat], dim=-1))

        pose_delta = 0.1 * torch.tanh(self.pose_delta_head(fused))
        intr_delta = 0.05 * torch.tanh(self.intr_delta_head(fused))

        fx = intrinsics[:, 0] * (1.0 + intr_delta[:, 0])
        fy = intrinsics[:, 1] * (1.0 + intr_delta[:, 1])
        cx = intrinsics[:, 2] + intr_delta[:, 2]
        cy = intrinsics[:, 3] + intr_delta[:, 3]
        intrinsics_corr = torch.stack([fx, fy, cx, cy], dim=-1)

        cam_t = camera_pose[:, :3] + pose_delta[:, :3]
        cam_rpy = camera_pose[:, 3:] + pose_delta[:, 3:]
        r_cw = _rpy_to_matrix(cam_rpy)

        pos = self.pos_head(fused)
        pos_z = F.softplus(pos[:, 2:3]) + self.min_depth
        pos = torch.cat([pos[:, :2], pos_z], dim=-1)
        vel = self.vel_head(fused)
        acc_seq = self.acc_head(fused).view(-1, self.future_steps, 3)

        points_cam: List[torch.Tensor] = []
        p_w = pos
        v_w = vel
        for step in range(self.future_steps):
            dt_s = dt[:, step : step + 1]
            a = acc_seq[:, step]
            v_w = v_w + a * dt_s
            p_w = p_w + v_w * dt_s
            rel_w = p_w - cam_t
            rel_c = torch.bmm(r_cw, rel_w.unsqueeze(-1)).squeeze(-1)
            points_cam.append(rel_c)

        points_cam_t = torch.stack(points_cam, dim=1)
        x = points_cam_t[:, :, 0]
        y = points_cam_t[:, :, 1]
        z = points_cam_t[:, :, 2].clamp_min(self.min_depth)

        u = fx.unsqueeze(1) * (x / z) + cx.unsqueeze(1)
        v = fy.unsqueeze(1) * (y / z) + cy.unsqueeze(1)
        pred_centers = torch.stack([u, v], dim=-1)

        return {
            "pred_centers": pred_centers,
            "pose_delta": pose_delta,
            "intrinsics_delta": intr_delta,
            "dynamics_pos0": pos,
            "dynamics_vel0": vel,
            "dynamics_acc": acc_seq,
            "intrinsics_corrected": intrinsics_corr,
        }
