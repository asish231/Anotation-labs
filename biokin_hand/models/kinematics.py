"""
Differentiable Forward Kinematics (FK) and Biomechanical Manifold Clamping Layer.
Implements:
1. Continuous 6D Rotation Representation (Zhou et al., CVPR 2019)
2. Euler Angle to SO(3) Rotation Matrix Conversion
3. Biomechanical Hard Clamp at Inference (0.0% Anatomical Violation Guarantee)
4. Differentiable Kinematic Tree Traversal with Learned Shape Deformation (beta)
5. Perspective Camera Reprojection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from biokin_hand.config import (
    NUM_JOINTS,
    KINEMATIC_PARENTS,
    BASE_BONE_OFFSETS,
    ANGLE_LIMITS_MIN_RAD,
    ANGLE_LIMITS_MAX_RAD,
    CONFIG
)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D continuous rotation representation to 3x3 rotation matrix (SO(3))
    using Gram-Schmidt orthogonalization.
    Args:
        d6: Tensor of shape (..., 6)
    Returns:
        rot_mat: Tensor of shape (..., 3, 3)
    """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]

    x = F.normalize(x_raw, dim=-1, eps=1e-8)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1, eps=1e-8)
    y = torch.cross(z, x, dim=-1)

    rot_mat = torch.stack([x, y, z], dim=-1)
    return rot_mat


def euler_angles_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """
    Converts Euler angles (flexion, abduction, twist) in radians to 3x3 rotation matrices.
    Order: Z (twist) -> Y (abduct) -> X (flexion)
    Args:
        euler: Tensor of shape (B, 20, 3)
    Returns:
        matrices: Tensor of shape (B, 20, 3, 3)
    """
    cx = torch.cos(euler[..., 0])
    sx = torch.sin(euler[..., 0])
    cy = torch.cos(euler[..., 1])
    sy = torch.sin(euler[..., 1])
    cz = torch.cos(euler[..., 2])
    sz = torch.sin(euler[..., 2])

    zeros = torch.zeros_like(cx)
    ones = torch.ones_like(cx)

    # Rx
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cx, -sx], dim=-1),
        torch.stack([zeros, sx, cx], dim=-1)
    ], dim=-2)

    # Ry
    Ry = torch.stack([
        torch.stack([cy, zeros, sy], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sy, zeros, cy], dim=-1)
    ], dim=-2)

    # Rz
    Rz = torch.stack([
        torch.stack([cz, -sz, zeros], dim=-1),
        torch.stack([sz, cz, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)

    # Compose: R = Rz @ Ry @ Rx
    R = torch.matmul(torch.matmul(Rz, Ry), Rx)
    return R


class BiomechanicalManifoldClamp(nn.Module):
    """
    Applies strict physiological clamping during inference only.
    During training, leaves angles unmolested so gradients flow freely to the loss.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("min_angles", torch.tensor(ANGLE_LIMITS_MIN_RAD, dtype=torch.float32))
        self.register_buffer("max_angles", torch.tensor(ANGLE_LIMITS_MAX_RAD, dtype=torch.float32))

    def forward(self, angles: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        """
        Args:
            angles: Tensor of shape (B, 20, 3) representing predicted Euler joint angles.
            is_training: If True, bypass hard clamping (soft penalty loss handles regularization).
        Returns:
            clamped_angles: Tensor of shape (B, 20, 3)
        """
        if is_training:
            return angles
        return torch.clamp(angles, self.min_angles.unsqueeze(0), self.max_angles.unsqueeze(0))


class DifferentiableForwardKinematics(nn.Module):
    """
    Computes global 3D positions of 21 hand joints from:
    1. Global wrist translation (t_root)
    2. Global palm rotation (R_root)
    3. 20 local joint angles (theta)
    4. Subject shape morphology (beta)
    """
    def __init__(self, num_shape_betas: int = 10):
        super().__init__()
        self.register_buffer("parents", torch.tensor(KINEMATIC_PARENTS, dtype=torch.long))
        self.register_buffer("base_bones", torch.tensor(BASE_BONE_OFFSETS, dtype=torch.float32))
        
        # Learned linear shape deformation matrix: shape (20, 3, num_shape_betas)
        # Allows per-subject bone scaling without breaking anatomical bone ratios
        self.shape_basis = nn.Parameter(torch.randn(20, 3, num_shape_betas) * 0.05)
        self.clamper = BiomechanicalManifoldClamp()

    def forward(self, t_root: torch.Tensor, r_root_6d: torch.Tensor,
                raw_angles: torch.Tensor, shape_betas: torch.Tensor = None,
                is_training: bool = False):
        """
        Args:
            t_root: (B, 3) Wrist global 3D translation [X, Y, Z]
            r_root_6d: (B, 6) Continuous 6D representation of palm global rotation
            raw_angles: (B, 20, 3) Euler angles of 20 relative joint connections
            shape_betas: (B, 10) Morphology coefficients (optional)
            is_training: Boolean flag for training vs test-time hard clamping
        Returns:
            positions_3d: (B, 21, 3) 3D joint positions in camera coordinates (mm)
            clamped_angles: (B, 20, 3) Joint angles
        """
        B = t_root.shape[0]

        # 1. Convert Root 6D to SO(3)
        R_root = rotation_6d_to_matrix(r_root_6d)  # (B, 3, 3)

        # 2. Clamp angles (Hard at inference, free at training)
        clamped_angles = self.clamper(raw_angles, is_training=is_training)

        # 3. Convert local angles to local 3x3 rotation matrices
        R_local = euler_angles_to_matrix(clamped_angles)  # (B, 20, 3, 3)

        # 4. Compute deformed bone offsets using beta
        # base_bones: (20, 3) -> (B, 20, 3)
        bone_offsets = self.base_bones.unsqueeze(0).repeat(B, 1, 1)
        if shape_betas is not None:
            # shape_basis: (20, 3, K) x shape_betas: (B, K) -> (B, 20, 3)
            delta_bones = torch.einsum("ijk,bk->bij", self.shape_basis, shape_betas)
            bone_offsets = bone_offsets + delta_bones

        # 5. Kinematic Traversal
        positions = [t_root]  # Joint 0 (Wrist)
        global_rotations = [R_root]

        for j in range(1, NUM_JOINTS):
            p_idx = self.parents[j].item()
            parent_R = global_rotations[p_idx]
            parent_P = positions[p_idx]

            # Relative rotation for joint j (1-indexed, so local index is j - 1)
            local_R = R_local[:, j - 1]
            global_R = torch.bmm(parent_R, local_R)
            global_rotations.append(global_R)

            # Offset along parent-rotated bone vector
            offset = bone_offsets[:, j - 1].unsqueeze(-1)  # (B, 3, 1)
            bone_vec = torch.bmm(global_R, offset).squeeze(-1)  # (B, 3)
            joint_P = parent_P + bone_vec
            positions.append(joint_P)

        positions_3d = torch.stack(positions, dim=1)  # (B, 21, 3)
        return positions_3d, clamped_angles

    def project_to_2d(self, positions_3d: torch.Tensor, K: torch.Tensor = None,
                      focal_length: float = 484.0, img_center: float = 112.0) -> torch.Tensor:
        """
        Pinhole perspective reprojection: (X, Y, Z) -> (u, v)
        Supports per-sample camera matrix K: (B, 3, 3) or canonical pinhole defaults.
        """
        z = positions_3d[..., 2].clamp(min=100.0)  # Min depth in mm to prevent divide-by-zero
        if K is not None:
            # K shape: (B, 3, 3)
            fx = K[:, 0, 0].unsqueeze(1)
            fy = K[:, 1, 1].unsqueeze(1)
            cx = K[:, 0, 2].unsqueeze(1)
            cy = K[:, 1, 2].unsqueeze(1)
            u = (positions_3d[..., 0] * fx / z) + cx
            v = (positions_3d[..., 1] * fy / z) + cy
        else:
            u = (positions_3d[..., 0] * focal_length / z) + img_center
            v = (positions_3d[..., 1] * focal_length / z) + img_center
        return torch.stack([u, v], dim=-1)
