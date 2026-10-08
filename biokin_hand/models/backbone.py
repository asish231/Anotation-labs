"""
Vision Backbone Module.
Wraps DINOv2-Small (Vision Transformer with 14x14 patch tokens)
with flexible fallback to Swin-V2 or TorchVision ViT.
Outputs:
1. Dense spatial patch tokens: (B, num_patches, embed_dim)
2. Global pooled embedding: (B, embed_dim)
"""
import torch
import torch.nn as nn
import timm

from biokin_hand.config import CONFIG


class VisionBackbone(nn.Module):
    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m",
                 pretrained: bool = True, output_dim: int = 384):
        super().__init__()
        self.output_dim = output_dim

        try:
            # Primary: DINOv2 via timm
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,       # Remove classifier head
                dynamic_img_size=True
            )
            raw_dim = self.model.embed_dim
        except Exception as e:
            print(f"[VisionBackbone] Note: Loading fallback backbone due to: {e}")
            # Fallback: TorchVision Swin-V2-Tiny
            from torchvision.models import swin_v2_t, Swin_V2_T_Weights
            weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
            self.model = swin_v2_t(weights=weights)
            raw_dim = self.model.head.in_features
            self.model.head = nn.Identity()

        if raw_dim != output_dim:
            self.proj = nn.Linear(raw_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (B, 3, H, W) normalized to [0, 1] or ImageNet norm
        Returns:
            patch_tokens: (B, N, output_dim) dense spatial features
            global_feat: (B, output_dim) global summary vector
        """
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if hasattr(self.model, "forward_head"):
                global_feat = self.model.forward_head(feats, pre_logits=True)
            elif isinstance(feats, tuple):
                global_feat = feats[0]
            elif feats.dim() == 3:
                global_feat = feats[:, 0]  # CLS token
            else:
                global_feat = feats.mean(dim=[-2, -1])

            if isinstance(feats, torch.Tensor) and feats.dim() == 3:
                # Exclude CLS token if present (tokens: 1 to end)
                if feats.shape[1] > 1:
                    patch_tokens = feats[:, 1:]
                else:
                    patch_tokens = feats
            else:
                # If 4D tensor (B, C, H, W)
                B, C, H, W = feats.shape
                patch_tokens = feats.flatten(2).transpose(1, 2)
        else:
            # Fallback forward pass
            global_feat = self.model(x)
            patch_tokens = global_feat.unsqueeze(1)

        patch_tokens = self.proj(patch_tokens)
        global_feat = self.proj(global_feat)
        return patch_tokens, global_feat
