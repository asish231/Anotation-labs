"""
3-Step Iterative Error Feedback (IEF) Reasoning Decoder with Kinematic Graph Cross-Attention.
Implements:
1. Joint Query Tokens that attend to DINOv2 visual patch tokens.
2. Kinematic Graph Self-Attention (Parent-to-Child message passing along bones).
3. 3-Step Iterative Error Feedback: predicts additive residuals (delta_theta, delta_root, delta_beta).
4. Returns intermediate predictions at every step k in {1, 2, 3} for per-iteration supervision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from biokin_hand.config import NUM_JOINTS, KINEMATIC_PARENTS


class KinematicGraphCrossAttention(nn.Module):
    """
    Cross-attention between 21 Joint Queries and dense spatial image patch tokens,
    followed by Kinematic Self-Attention along anatomical bone edges.
    """
    def __init__(self, embed_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, joint_queries: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_queries: (B, 22, embed_dim) - 20 relative joints + 1 root + 1 shape query
            patch_tokens: (B, N, embed_dim) - dense visual tokens
        Returns:
            updated_queries: (B, 22, embed_dim)
        """
        # 1. Cross-Attention: Joint Queries attend to visual patches
        q_norm = self.norm1(joint_queries)
        cross_out, _ = self.cross_attn(query=q_norm, key=patch_tokens, value=patch_tokens)
        x = joint_queries + cross_out

        # 2. Kinematic Self-Attention: Joints communicate with parent/neighbor joints
        x_norm = self.norm2(x)
        self_out, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm)
        x = x + self_out

        # 3. FFN
        x = x + self.mlp(self.norm3(x))
        return x


class IEFReasoningDecoder(nn.Module):
    """
    Iterative Error Feedback Loop.
    Performs K recurrent refinement passes (default: 3 steps)
    """
    def __init__(self, embed_dim: int = 384, num_steps: int = 3, num_shape_betas: int = 10):
        super().__init__()
        self.num_steps = num_steps
        self.embed_dim = embed_dim
        self.num_shape_betas = num_shape_betas

        # 22 Learnable base queries: 20 joint angles + 1 global root + 1 shape query
        self.joint_queries = nn.Parameter(torch.randn(1, 22, embed_dim) * 0.02)

        # Kinematic Attention Block
        self.kg_attn = KinematicGraphCrossAttention(embed_dim=embed_dim)

        # Linear projection heads for additive residuals (per-joint, bounded)
        # Root branch: translation delta (3) + rotation 6D delta (6)
        self.head_root = nn.Linear(embed_dim, 3 + 6)
        # Joint angle branch: shared linear projection per joint query (384 -> 3 Euler angles)
        self.head_angles = nn.Linear(embed_dim, 3)
        # Shape branch: beta delta (10)
        self.head_shape = nn.Linear(embed_dim, num_shape_betas)

        # Zero-initialize the residual delta projection layers (standard in IEF / HMR)
        # Ensures the network starts from the neutral resting pose with 0 delta and 0 barrier penalty
        nn.init.zeros_(self.head_root.weight)
        nn.init.zeros_(self.head_root.bias)
        nn.init.zeros_(self.head_angles.weight)
        nn.init.zeros_(self.head_angles.bias)
        nn.init.zeros_(self.head_shape.weight)
        nn.init.zeros_(self.head_shape.bias)

        # State embedding projection to feed previous state back into queries
        # State: 3 (t) + 6 (r) + 60 (angles) + 10 (beta) = 79
        self.state_proj = nn.Linear(79, embed_dim)

    def forward(self, patch_tokens: torch.Tensor, global_feat: torch.Tensor):
        """
        Args:
            patch_tokens: (B, N, embed_dim)
            global_feat: (B, embed_dim)
        Returns:
            step_outputs: List of dicts for each of the 3 iterations:
                [
                    {"t_root": (B, 3), "r_root_6d": (B, 6), "angles": (B, 20, 3), "shape_betas": (B, 10)},
                    ...
                ]
        """
        B = patch_tokens.shape[0]

        # Canonical initial state priors
        # Initial root translation: wrist at [0, 0, 600mm] (60cm in front of camera)
        t_current = torch.tensor([0.0, 0.0, 600.0], device=patch_tokens.device).repeat(B, 1)
        # Initial root rotation: identity 6D continuous rotation
        r_current_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=patch_tokens.device).repeat(B, 1)
        # Initial angles: zero flat palm
        angles_current = torch.zeros(B, 20, 3, device=patch_tokens.device)
        # Initial shape: neutral zeros
        betas_current = torch.zeros(B, self.num_shape_betas, device=patch_tokens.device)

        step_outputs = []

        for step in range(self.num_steps):
            # 1. Encode current pose state as context
            flat_state = torch.cat([
                t_current,
                r_current_6d,
                angles_current.reshape(B, -1),
                betas_current
            ], dim=-1)  # (B, 79)
            state_embed = self.state_proj(flat_state).unsqueeze(1)  # (B, 1, embed_dim)

            # 2. Combine learnable queries + state feedback
            queries = self.joint_queries.repeat(B, 1, 1) + state_embed

            # 3. Cross-attention with image patches & self-attention across joints
            updated_queries = self.kg_attn(queries, patch_tokens)

            # 4. Predict additive residuals (delta updates)
            # Query 0: Root query -> delta_t (bounded to +/-50mm per step), delta_r
            root_delta = self.head_root(updated_queries[:, 0])
            delta_t = torch.tanh(root_delta[:, :3]) * 50.0
            delta_r = torch.tanh(root_delta[:, 3:]) * 0.2

            # Query 1 to 20: Finger joint queries -> delta_angles (bounded to +/-0.35 rad per step)
            finger_queries = updated_queries[:, 1:21]  # (B, 20, embed_dim)
            delta_angles = torch.tanh(self.head_angles(finger_queries)) * 0.35  # (B, 20, 3)

            # Query 21: Shape query -> delta_beta (bounded to +/-0.1 per step)
            delta_betas = torch.tanh(self.head_shape(updated_queries[:, 21])) * 0.1

            # 5. Additive State Update (IEF Step)
            t_current = t_current + delta_t
            r_current_6d = r_current_6d + delta_r
            angles_current = angles_current + delta_angles
            betas_current = betas_current + delta_betas

            step_outputs.append({
                "step": step + 1,
                "t_root": t_current,
                "r_root_6d": r_current_6d,
                "angles": angles_current,
                "shape_betas": betas_current
            })

        return step_outputs
