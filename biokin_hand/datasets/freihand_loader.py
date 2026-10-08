"""
FreiHAND Dataset Loader for BioKin-Transformer.
Parses the official FreiHAND v2 dataset:
- training/rgb/: 32,560 original images (up to 130k with augmented backgrounds)
- training_xyz.json: Ground-truth 3D joint coordinates (in meters)
- training_k.json: Camera intrinsic matrices (3x3)
- training_mano.json: Ground-truth MANO angles (theta) and shape (beta)

Directly maps FreiHAND ground-truth annotations to BioKinNet's Kinematic representation.
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from biokin_hand.config import CONFIG, NUM_JOINTS


def project_3d_to_2d(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Projects 3D keypoints (meters) to 2D image coordinates using camera intrinsics K.
    Args:
        xyz: (21, 3) 3D coordinates in meters
        K: (3, 3) Camera intrinsic matrix
    Returns:
        uv: (21, 2) 2D pixel coordinates
    """
    uv_homo = np.matmul(K, xyz.T).T  # (21, 3)
    u = uv_homo[:, 0] / np.maximum(uv_homo[:, 2], 1e-6)
    v = uv_homo[:, 1] / np.maximum(uv_homo[:, 2], 1e-6)
    return np.stack([u, v], axis=-1)


class FreiHANDDataset(Dataset):
    def __init__(self, data_root: str = "./biokin_data/freihand",
                 split: str = "training", img_size: int = CONFIG["image_size"],
                 max_samples: int = None):
        """
        Args:
            data_root: Path containing the unzipped FreiHAND dataset
            split: 'training' or 'evaluation'
            img_size: Canonical image size for DINOv2 (default: 224)
            max_samples: Optional limit for fast experimentation / debugging
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size

        # Paths
        self.rgb_dir = os.path.join(data_root, f"{split}/rgb")
        xyz_file = os.path.join(data_root, f"{split}_xyz.json")
        k_file = os.path.join(data_root, f"{split}_k.json")
        if not os.path.exists(k_file):
            k_file = os.path.join(data_root, f"{split}_K.json")
        mano_file = os.path.join(data_root, f"{split}_mano.json")

        if not os.path.exists(xyz_file) or not os.path.exists(k_file):
            print(f"[FreiHANDDataset] Warning: Annotation files not found in {data_root}.")
            print(f"Make sure FreiHAND_pub_v2.zip is extracted inside {data_root}.")
            self.xyz_list = []
            self.k_list = []
            self.mano_list = []
            self.num_samples = 0
            return

        print(f"[FreiHANDDataset] Loading annotations from {data_root}...")
        with open(xyz_file, "r") as f:
            self.xyz_list = json.load(f)
        with open(k_file, "r") as f:
            self.k_list = json.load(f)

        if os.path.exists(mano_file):
            with open(mano_file, "r") as f:
                self.mano_list = json.load(f)
        else:
            self.mano_list = None

        self.num_samples = len(self.xyz_list)
        if max_samples is not None:
            self.num_samples = min(self.num_samples, max_samples)

        print(f"[FreiHANDDataset] Successfully indexed {self.num_samples} FreiHAND samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Image filename: 8-digit zero-padded index, e.g. 00000000.jpg
        img_name = f"{idx:08d}.jpg"
        img_path = os.path.join(self.rgb_dir, img_name)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback black canvas if image missing
            img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
            orig_h, orig_w = 224, 224
        else:
            orig_h, orig_w = img_bgr.shape[:2]

        # 3D metric keypoints in meters -> convert to mm (scale x 1000)
        xyz_m = np.array(self.xyz_list[idx], dtype=np.float32)  # (21, 3) in meters
        xyz_mm = xyz_m * 1000.0                                 # (21, 3) in millimeters

        # Camera intrinsics (3, 3)
        K = np.array(self.k_list[idx], dtype=np.float32)

        # 2D projected keypoints (original image pixel coords)
        uv_orig = project_3d_to_2d(xyz_m, K)                    # (21, 2)

        # Resize image to target img_size (224x224)
        resized_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Scale 2D keypoints to target img_size
        uv_scaled = uv_orig.copy()
        uv_scaled[:, 0] = (uv_scaled[:, 0] / max(1.0, orig_w)) * self.img_size
        uv_scaled[:, 1] = (uv_scaled[:, 1] / max(1.0, orig_h)) * self.img_size

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1)     # (3, 224, 224)
        target_2d = torch.from_numpy(uv_scaled.astype(np.float32))
        target_3d = torch.from_numpy(xyz_mm.astype(np.float32))
        k_tensor = torch.from_numpy(K.astype(np.float32))

        # MANO parameters (if available)
        sample = {
            "image": img_tensor,
            "target_2d": target_2d,
            "target_3d": target_3d,
            "camera_k": k_tensor,
            "img_path": img_path
        }

        if self.mano_list is not None and idx < len(self.mano_list):
            mano_raw = self.mano_list[idx]
            vec = mano_raw[0] if (isinstance(mano_raw, list) and len(mano_raw) > 0) else mano_raw
            if isinstance(vec, list) and len(vec) >= 58:
                sample["mano_pose"] = torch.tensor(vec[:48], dtype=torch.float32)
                sample["mano_shape"] = torch.tensor(vec[48:58], dtype=torch.float32)

        return sample
