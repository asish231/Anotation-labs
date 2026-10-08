"""
Biomechanical Loss Module.
Implements:
1. 2D Reprojection Loss (Smooth L1 / Huber)
2. 3D Metric Joint Euclidean Loss
3. Soft Barrier Biomechanical Angle Penalty (Solves Sigmoid Gradient Saturation)
4. Shape Morphology L2 Regularization
5. Multi-Step IEF Supervision Discounting (gamma^(3-k))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from biokin_hand.config import (
    ANGLE_LIMITS_MIN_RAD,
    ANGLE_LIMITS_MAX_RAD,
    CONFIG
)


class BiomechanicalLoss(nn.Module):
    def __init__(self, lambda_2d: float = None, lambda_3d: float = None,
                 lambda_root: float = None, lambda_bio: float = None,
                 lambda_shape: float = None, step_decay_gamma: float = None):
        super().__init__()
        weights = CONFIG["loss_weights"]
        self.lambda_2d = lambda_2d if lambda_2d is not None else weights["lambda_2d"]
        self.lambda_3d = lambda_3d if lambda_3d is not None else weights["lambda_3d"]
        self.lambda_root = lambda_root if lambda_root is not None else weights.get("lambda_root", 20.0)
        self.lambda_bio = lambda_bio if lambda_bio is not None else weights["lambda_bio"]
        self.lambda_shape = lambda_shape if lambda_shape is not None else weights["lambda_shape"]
        self.gamma = step_decay_gamma if step_decay_gamma is not None else weights["step_decay_gamma"]

        # Buffers for angle bounds
        self.register_buffer("min_angles", torch.tensor(ANGLE_LIMITS_MIN_RAD, dtype=torch.float32))
        self.register_buffer("max_angles", torch.tensor(ANGLE_LIMITS_MAX_RAD, dtype=torch.float32))

    def compute_bio_penalty(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Soft quadratic barrier function.
        Gradients are zero when inside limits, and grow smoothly when limits are violated.
        """
        min_viol = F.relu(self.min_angles.unsqueeze(0) - angles)  # angles < min
        max_viol = F.relu(angles - self.max_angles.unsqueeze(0))  # angles > max
        penalty = (min_viol ** 2 + max_viol ** 2).sum(dim=[-1, -2]).mean()
        return penalty

    def forward(self, step_predictions: list, target_2d: torch.Tensor,
                target_3d: torch.Tensor = None, visibility_mask: torch.Tensor = None):
        """
        Args:
            step_predictions: List of outputs from BioKinNet in training mode (len=3)
            target_2d: (B, 21, 2) Ground truth 2D keypoints (pixels, 0..224)
            target_3d: (B, 21, 3) Ground truth 3D coordinates in millimeters
            visibility_mask: (B, 21) Boolean or float visibility weights
        Returns:
            loss_dict: Dictionary of individual losses and total scalar loss
        """
        total_loss = 0.0
        num_steps = len(step_predictions)

        loss_2d_acc = 0.0
        loss_3d_acc = 0.0
        loss_bio_acc = 0.0
        loss_shape_acc = 0.0

        for k, out in enumerate(step_predictions):
            discount = self.gamma ** (num_steps - 1 - k)
            proj_2d = out["projected_2d"]
            pos_3d = out["positions_3d"]
            angles = out["angles"]
            betas = out["shape_betas"]

            # 1. 2D Reprojection Loss (Huber normalized to fraction of image size 224)
            if visibility_mask is not None:
                mask = visibility_mask.unsqueeze(-1)
                l2d = F.smooth_l1_loss(proj_2d * mask, target_2d * mask, reduction="sum") / (mask.sum() * 2 + 1e-6) / 224.0
            else:
                l2d = F.smooth_l1_loss(proj_2d, target_2d) / 224.0

            # 2. 3D Metric Loss in METERS (pos_3d and target_3d are in mm -> divide by 1000)
            if target_3d is not None:
                # Root-relative joint distances in meters
                pred_root_rel_m = (pos_3d - pos_3d[:, :1]) / 1000.0
                gt_root_rel_m = (target_3d - target_3d[:, :1]) / 1000.0
                l3d = F.l1_loss(pred_root_rel_m, gt_root_rel_m)

                # Wrist global position in meters
                pred_root_m = pos_3d[:, 0] / 1000.0
                gt_root_m = target_3d[:, 0] / 1000.0
                l_root = F.l1_loss(pred_root_m, gt_root_m)
            else:
                l3d = torch.tensor(0.0, device=angles.device)
                l_root = torch.tensor(0.0, device=angles.device)

            # 3. Soft Biomechanical Angle Penalty
            lbio = self.compute_bio_penalty(angles)

            # 4. Shape regularization (prevent abnormal bone scales)
            lshape = (betas ** 2).mean()

            step_loss = (
                self.lambda_2d * l2d +
                self.lambda_3d * l3d +
                self.lambda_root * l_root +
                self.lambda_bio * lbio +
                self.lambda_shape * lshape
            )

            total_loss += discount * step_loss

            # Track 2D in actual average pixel error for human interpretability
            loss_2d_acc += (F.l1_loss(proj_2d, target_2d)).item()
            loss_3d_acc += (l3d.item() * 1000.0) if isinstance(l3d, torch.Tensor) else (l3d * 1000.0) # in mm
            loss_bio_acc += lbio.item()
            loss_shape_acc += lshape.item()

        return {
            "loss": total_loss,
            "loss_2d_px": loss_2d_acc / num_steps,
            "loss_3d_mm": loss_3d_acc / num_steps,
            "loss_bio": loss_bio_acc / num_steps,
            "loss_shape": loss_shape_acc / num_steps
        }
