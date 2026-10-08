"""
BioKin-Transformer: Canonical Configuration & Anatomical Kinematic Specification.
Defines:
- 21-Joint Skeletal Graph Tree
- Kinematic Parent-Child Indices
- Canonical Anthropomorphic Bone Template Offsets
- Physiological Joint Angle Limits (in Radians and Degrees)
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# 1. Kinematic Graph Topology (21 Joints)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "Wrist",
    "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"
]

NUM_JOINTS = 21

# Parent index for each joint (0 is root Wrist)
KINEMATIC_PARENTS = [
    0,   # 0: Wrist (Root)
    0, 1, 2, 3,       # 1-4: Thumb
    0, 5, 6, 7,       # 5-8: Index
    0, 9, 10, 11,     # 9-12: Middle
    0, 13, 14, 15,    # 13-16: Ring
    0, 17, 18, 19     # 17-20: Pinky
]

FINGER_CHAINS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20]
}

# ---------------------------------------------------------------------------
# 2. Canonical Anthropometric Bone Offsets (in mm, rest T-pose / flat palm)
# Measured relative to parent joint frame in canonical 3D coordinates.
# ---------------------------------------------------------------------------
BASE_BONE_OFFSETS = [
    # Joint 1-4: Thumb (CMC, MCP, IP, Tip)
    [-24.0, -18.0, 10.0],
    [-20.0, -26.0, 12.0],
    [-16.0, -22.0, 10.0],
    [-14.0, -18.0, 8.0],
    # Joint 5-8: Index (MCP, PIP, DIP, Tip)
    [-18.0, -68.0, 4.0],
    [-3.0, -38.0, 0.0],
    [-2.0, -24.0, 0.0],
    [-1.0, -18.0, 0.0],
    # Joint 9-12: Middle (MCP, PIP, DIP, Tip)
    [0.0, -72.0, 0.0],
    [0.0, -42.0, 0.0],
    [0.0, -26.0, 0.0],
    [0.0, -20.0, 0.0],
    # Joint 13-16: Ring (MCP, PIP, DIP, Tip)
    [16.0, -66.0, -3.0],
    [2.0, -38.0, 0.0],
    [1.0, -24.0, 0.0],
    [1.0, -18.0, 0.0],
    # Joint 17-20: Pinky (MCP, PIP, DIP, Tip)
    [30.0, -58.0, -6.0],
    [4.0, -30.0, 0.0],
    [3.0, -18.0, 0.0],
    [2.0, -15.0, 0.0],
]

# ---------------------------------------------------------------------------
# 3. Biomechanical Joint Limits (Min & Max in Radians)
# For each of the 20 relative joint connections (joints 1 to 20):
# Angles represent (Flexion/Extension, Abduction/Adduction, Pronation/Supination)
# ---------------------------------------------------------------------------
def deg2rad(d):
    return d * math.pi / 180.0

# [min_flex, max_flex, min_abduct, max_abduct, min_twist, max_twist]
# All non-zero knuckle hinge joints CANNOT bend backward (min_flex >= 0)
JOINT_LIMITS_DEG = {
    # Thumb CMC (Saddle joint with opposition)
    1: {"flex": (-20.0, 70.0), "abduct": (0.0, 60.0), "twist": (-15.0, 15.0)},
    # Thumb MCP
    2: {"flex": (-10.0, 70.0), "abduct": (-10.0, 10.0), "twist": (-5.0, 5.0)},
    # Thumb IP (Hinge: zero backward bend)
    3: {"flex": (-5.0, 90.0), "abduct": (-3.0, 3.0), "twist": (-2.0, 2.0)},
    # Thumb Tip (End effector fixed)
    4: {"flex": (0.0, 0.0), "abduct": (0.0, 0.0), "twist": (0.0, 0.0)},

    # Index MCP (Knuckle)
    5: {"flex": (-15.0, 90.0), "abduct": (-20.0, 20.0), "twist": (-8.0, 8.0)},
    # Index PIP (Strict 1-DoF hinge: cannot bend backwards)
    6: {"flex": (0.0, 100.0), "abduct": (-3.0, 3.0), "twist": (-2.0, 2.0)},
    # Index DIP (Strict 1-DoF hinge: cannot bend backwards)
    7: {"flex": (0.0, 90.0), "abduct": (-2.0, 2.0), "twist": (-2.0, 2.0)},
    # Index Tip
    8: {"flex": (0.0, 0.0), "abduct": (0.0, 0.0), "twist": (0.0, 0.0)},

    # Middle MCP
    9: {"flex": (-15.0, 90.0), "abduct": (-15.0, 15.0), "twist": (-5.0, 5.0)},
    # Middle PIP
    10: {"flex": (0.0, 100.0), "abduct": (-2.0, 2.0), "twist": (-2.0, 2.0)},
    # Middle DIP
    11: {"flex": (0.0, 90.0), "abduct": (-2.0, 2.0), "twist": (-2.0, 2.0)},
    # Middle Tip
    12: {"flex": (0.0, 0.0), "abduct": (0.0, 0.0), "twist": (0.0, 0.0)},

    # Ring MCP
    13: {"flex": (-15.0, 90.0), "abduct": (-15.0, 15.0), "twist": (-5.0, 5.0)},
    # Ring PIP
    14: {"flex": (0.0, 100.0), "abduct": (-2.0, 2.0), "twist": (-2.0, 2.0)},
    # Ring DIP
    15: {"flex": (0.0, 90.0), "abduct": (-2.0, 2.0), "twist": (-2.0, 2.0)},
    # Ring Tip
    16: {"flex": (0.0, 0.0), "abduct": (0.0, 0.0), "twist": (0.0, 0.0)},

    # Pinky MCP
    17: {"flex": (-15.0, 90.0), "abduct": (-25.0, 25.0), "twist": (-10.0, 10.0)},
    # Pinky PIP
    18: {"flex": (0.0, 100.0), "abduct": (-3.0, 3.0), "twist": (-2.0, 2.0)},
    # Pinky DIP
    19: {"flex": (0.0, 90.0), "abduct": (-2.0, 2.0), "twist": (-2.0, 2.0)},
    # Pinky Tip
    20: {"flex": (0.0, 0.0), "abduct": (0.0, 0.0), "twist": (0.0, 0.0)},
}

# Compile into flat PyTorch-ready limit matrices (20 joints x 3 Euler axes)
ANGLE_LIMITS_MIN_RAD = np.zeros((20, 3), dtype=np.float32)
ANGLE_LIMITS_MAX_RAD = np.zeros((20, 3), dtype=np.float32)

for j_idx in range(1, 21):
    cfg = JOINT_LIMITS_DEG[j_idx]
    row = j_idx - 1
    ANGLE_LIMITS_MIN_RAD[row, 0] = deg2rad(cfg["flex"][0])
    ANGLE_LIMITS_MAX_RAD[row, 0] = deg2rad(cfg["flex"][1])
    ANGLE_LIMITS_MIN_RAD[row, 1] = deg2rad(cfg["abduct"][0])
    ANGLE_LIMITS_MAX_RAD[row, 1] = deg2rad(cfg["abduct"][1])
    ANGLE_LIMITS_MIN_RAD[row, 2] = deg2rad(cfg["twist"][0])
    ANGLE_LIMITS_MAX_RAD[row, 2] = deg2rad(cfg["twist"][1])

# ---------------------------------------------------------------------------
# 4. Training & Architecture Hyperparameters
# ---------------------------------------------------------------------------
CONFIG = {
    "image_size": 224,
    "patch_size": 14,
    "feature_dim": 384,          # DINOv2-Small embedding dim
    "ief_steps": 3,              # 3-step Iterative Error Feedback
    "num_shape_betas": 10,       # Number of learned morphology PCA components
    "loss_weights": {
        "lambda_2d": 50.0,        # Normalized 2D pixel reprojection loss
        "lambda_3d": 50.0,        # 3D metric loss in meters (L1)
        "lambda_root": 20.0,      # Global wrist translation loss in meters (L1)
        "lambda_bio": 5.0,        # Soft barrier loss weight
        "lambda_shape": 0.01,
        "step_decay_gamma": 0.8
    },
    "camera": {
        "focal_length": 484.0,   # FreiHAND canonical pinhole camera focal length (mean fx, fy)
        "img_center": 112.0      # Principal point (224 / 2 = 112.0)
    }
}
