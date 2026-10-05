"""Blender-independent motion loading and SMPL rotation math."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = {1, 2}

# SMPL body order: pelvis, then the 21 body_pose joints.
SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

SMPL_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int32,
)

# Proper rotation: SMPL (X right, Y up, Z depth) -> Blender (X right, Z up).
SMPL_TO_BLENDER = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MotionData:
    body_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    fps: float
    betas: np.ndarray | None = None
    static_confidence: np.ndarray | None = None
    source_name: str = ""
    left_hand_landmarks: np.ndarray | None = None
    right_hand_landmarks: np.ndarray | None = None
    left_hand_confidence: np.ndarray | None = None
    right_hand_confidence: np.ndarray | None = None
    hand_backend: str = ""

    @property
    def frame_count(self) -> int:
        return int(self.body_pose.shape[0])


def _numeric_array(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array")
    value = value.astype(np.float64, copy=False)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


def _numeric_scalar(value: np.ndarray, name: str) -> float:
    value = _numeric_array(value, name)
    if value.size != 1:
        raise ValueError(f"{name} must contain exactly one numeric value, got shape {value.shape}")
    return float(value.reshape(-1)[0])


def _string_scalar(value: np.ndarray, name: str) -> str:
    value = np.asarray(value)
    if value.size != 1 or value.dtype.kind not in "SU":
        raise ValueError(f"{name} must contain exactly one string value")
    item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} must contain a UTF-8 string") from exc
    return str(item)


def _frame_vectors(value: np.ndarray, name: str) -> np.ndarray:
    value = _numeric_array(value, name)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [F,3] or [1,F,3], got {value.shape}")
    return value


def load_motion(path: str | Path) -> MotionData:
    """Load and validate a safe GVHMR motion NPZ (pickle is always disabled)."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("GVHMR Motion file must use the .npz extension")
    if not path.is_file():
        raise ValueError(f"Motion file not found: {path}")

    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Motion file must be a NumPy NPZ archive")
    with archive as data:
        required = {"format_version", "body_pose", "global_orient", "transl", "fps"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Missing NPZ fields: {', '.join(missing)}")

        version_value = _numeric_scalar(data["format_version"], "format_version")
        if not version_value.is_integer():
            raise ValueError("format_version must be an integer")
        version = int(version_value)
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported motion format version {version}; "
                f"expected one of {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )

        body_pose = _numeric_array(data["body_pose"], "body_pose")
        # Only an extra batch dimension may be removed. [1,21,3] is one frame.
        if body_pose.ndim == 4 and body_pose.shape[0] == 1:
            body_pose = body_pose[0]
        elif body_pose.ndim == 3 and body_pose.shape[0] == 1 and body_pose.shape[2] == 63:
            body_pose = body_pose[0]
        if body_pose.ndim == 2 and body_pose.shape[1] == 63:
            body_pose = body_pose.reshape(-1, 21, 3)
        if body_pose.ndim != 3 or body_pose.shape[1:] != (21, 3):
            raise ValueError(f"body_pose must have shape [F,21,3] or [F,63], got {body_pose.shape}")

        global_orient = _frame_vectors(data["global_orient"], "global_orient")
        transl = _frame_vectors(data["transl"], "transl")

        frame_count = body_pose.shape[0]
        if global_orient.shape[0] != frame_count or transl.shape[0] != frame_count:
            raise ValueError(
                "Frame count mismatch: "
                f"body_pose={frame_count}, global_orient={global_orient.shape[0]}, "
                f"transl={transl.shape[0]}"
            )
        if frame_count == 0:
            raise ValueError("Motion contains no frames")

        fps = _numeric_scalar(data["fps"], "fps")
        if not 1.0 <= fps <= 240.0:
            raise ValueError(f"Invalid FPS: {fps}")

        betas = None
        if "betas" in data.files:
            betas = _numeric_array(data["betas"], "betas").reshape(-1)
            if betas.size == 0:
                raise ValueError("betas must not be empty")

        static_confidence = None
        if "static_confidence" in data.files:
            static_confidence = _numeric_array(
                data["static_confidence"], "static_confidence"
            )
            if static_confidence.ndim == 3 and static_confidence.shape[0] == 1:
                static_confidence = static_confidence[0]
            if static_confidence.ndim not in (1, 2) or static_confidence.size == 0:
                raise ValueError("static_confidence must have shape [F] or [F,C]")
            if static_confidence.shape[0] != frame_count:
                raise ValueError("static_confidence frame count does not match body_pose")

        source_name = ""
        if "source_name" in data.files:
            source_name = _string_scalar(data["source_name"], "source_name")

        hand_values = {}
        for side in ("left", "right"):
            landmarks_key = f"{side}_hand_landmarks"
            confidence_key = f"{side}_hand_confidence"
            landmarks = confidence = None
            if confidence_key in data.files and landmarks_key not in data.files:
                raise ValueError(f"Missing NPZ field: {landmarks_key}")
            if landmarks_key in data.files:
                landmarks = _numeric_array(data[landmarks_key], landmarks_key)
                if landmarks.shape != (frame_count, 21, 3):
                    raise ValueError(
                        f"{landmarks_key} must have shape [F,21,3], got {landmarks.shape}"
                    )
                if confidence_key not in data.files:
                    raise ValueError(f"Missing NPZ field: {confidence_key}")
                confidence = _numeric_array(data[confidence_key], confidence_key)
                if confidence.shape != (frame_count,):
                    raise ValueError(
                        f"{confidence_key} must have shape [F], got {confidence.shape}"
                    )
                if np.any((confidence < 0.0) | (confidence > 1.0)):
                    raise ValueError(f"{confidence_key} must be in [0,1]")
            hand_values[f"{side}_hand_landmarks"] = landmarks
            hand_values[f"{side}_hand_confidence"] = confidence

        hand_backend = ""
        if "hand_backend" in data.files:
            hand_backend = _string_scalar(data["hand_backend"], "hand_backend")

    return MotionData(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        fps=fps,
        betas=betas,
        static_confidence=static_confidence,
        source_name=source_name,
        hand_backend=hand_backend,
        **hand_values,
    )


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues conversion for arrays ending in three components."""
    axis_angle = _numeric_array(axis_angle, "axis_angle")
    if axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle must end with three components")

    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe_theta = np.where(theta > 1e-12, theta, 1.0)
    axis = axis_angle / safe_theta
    x, y, z = np.moveaxis(axis, -1, 0)
    zeros = np.zeros_like(x)
    skew = np.stack(
        (
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ),
        axis=-1,
    ).reshape(axis_angle.shape[:-1] + (3, 3))

    identity = np.broadcast_to(np.eye(3), skew.shape)
    sin_theta = np.sin(theta)[..., None]
    cos_theta = np.cos(theta)[..., None]
    result = identity + sin_theta * skew + (1.0 - cos_theta) * (skew @ skew)

    # The Taylor limit avoids axis noise for exactly-zero rotations.
    small = (theta[..., 0] <= 1e-12)[..., None, None]
    return np.where(small, identity, result)


def smpl_global_rotations(motion: MotionData, *, flip_forward: bool = False) -> np.ndarray:
    """Return `[F,22,3,3]` rotations, optionally turning the whole motion.

    The heading flip is an armature-space rotation applied after the fixed
    SMPL-to-Blender basis conversion. Left-multiplying all global rotations
    turns the root while leaving parent-relative joint rotations unchanged.
    """
    local_axis_angle = np.concatenate(
        (motion.global_orient[:, None, :], motion.body_pose),
        axis=1,
    )
    local = axis_angle_to_matrix(local_axis_angle)
    global_smpl = np.empty_like(local)
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            global_smpl[:, joint] = local[:, joint]
        else:
            global_smpl[:, joint] = global_smpl[:, parent] @ local[:, joint]

    basis = SMPL_TO_BLENDER
    global_blender = basis @ global_smpl @ basis.T
    if flip_forward:
        flip = np.diag((-1.0, -1.0, 1.0))
        global_blender = flip @ global_blender
    return global_blender


def blender_translation(motion: MotionData, *, flip_forward: bool = False) -> np.ndarray:
    """Convert root translation to Blender axes and make frame zero the origin."""
    basis = SMPL_TO_BLENDER.copy()
    if flip_forward:
        basis = np.diag((-1.0, -1.0, 1.0)) @ basis
    translated = motion.transl @ basis.T
    return translated - translated[:1]
