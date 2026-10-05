"""Blender-independent, non-destructive motion preparation.

Smoothing uses a centered triangular window in source-video seconds. Rotation
windows average unit quaternions on the shortest hemisphere, avoiding the
axis-angle discontinuity at +/- pi. No extra dependency beyond NumPy is needed.
"""

from __future__ import annotations

from dataclasses import fields, replace
from numbers import Integral, Real

import numpy as np

from .core import MotionData

_FRAME_FIELDS = {
    "body_pose", "global_orient", "transl", "static_confidence",
    "left_hand_landmarks", "right_hand_landmarks",
    "left_hand_confidence", "right_hand_confidence",
}


def _number(value: float, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    value = float(value)
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    return value


def _radius(seconds: float, fps: float, frame_count: int) -> int:
    # Select the nearest odd-sized window, bounded by the available clip.
    return min(int(seconds * fps / 2.0 + 0.5), frame_count - 1)


def _window_slices(frame_count: int, offset: int) -> tuple[slice, slice]:
    if offset < 0:
        return slice(-offset, frame_count), slice(0, frame_count + offset)
    return slice(0, frame_count - offset), slice(offset, frame_count)


def _smooth_values(values: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return values.copy()
    result = np.zeros_like(values, dtype=np.float64)
    total_weight = np.zeros(len(values), dtype=np.float64)
    for offset in range(-radius, radius + 1):
        target, source = _window_slices(len(values), offset)
        weight = radius + 1 - abs(offset)
        result[target] += weight * values[source]
        total_weight[target] += weight
    return result / total_weight.reshape((-1,) + (1,) * (values.ndim - 1))


def _axis_angle_to_quaternion(values: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(values, axis=-1, keepdims=True)
    scale = np.empty_like(angle)
    np.divide(np.sin(angle / 2.0), angle, out=scale, where=angle > 1e-8)
    small = angle <= 1e-8
    scale[small] = 0.5 - angle[small] ** 2 / 48.0
    result = np.concatenate((np.cos(angle / 2.0), values * scale), axis=-1)
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    for frame in range(1, len(result)):
        signs = np.where(np.sum(result[frame] * result[frame - 1], axis=-1) < 0, -1, 1)
        result[frame] *= signs[..., None]
    return result


def _quaternion_to_axis_angle(values: np.ndarray) -> np.ndarray:
    # Canonicalize the quaternion so the returned angle remains in [0, pi].
    values = values * np.where(values[..., :1] < 0, -1.0, 1.0)
    vector = values[..., 1:]
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(length, np.clip(values[..., :1], 0.0, 1.0))
    scale = np.full_like(length, 2.0)
    np.divide(angle, length, out=scale, where=length > 1e-8)
    return vector * scale


def _smooth_rotations(values: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return values.copy()
    quaternions = _axis_angle_to_quaternion(values)
    result = np.zeros_like(quaternions)
    for offset in range(-radius, radius + 1):
        target, source = _window_slices(len(values), offset)
        neighbors = quaternions[source]
        # Align each neighbor to its window center before taking the weighted
        # quaternion mean. q and -q represent the same rotation.
        signs = np.where(np.sum(quaternions[target] * neighbors, axis=-1) < 0, -1.0, 1.0)
        result[target] += (radius + 1 - abs(offset)) * neighbors * signs[..., None]
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return _quaternion_to_axis_angle(result)


def prepare_motion(
    motion: MotionData,
    *,
    source_start: int = 1,
    source_end: int = 0,
    speed: float = 1.0,
    rotation_smoothing: float = 0.0,
    translation_smoothing: float = 0.0,
    root_motion: str = "FULL",
) -> MotionData:
    """Copy, trim, smooth, and retime a motion before creating Blender keys.

    Source frames are one-based and inclusive; source_end=0 selects the last
    frame. Speed changes playback duration via FPS while retaining every sample.
    Smoothing windows are 0..0.5 seconds at the original FPS; zero disables them.
    FULL retains root travel, IN_PLACE retains only vertical (SMPL Y) travel,
    and NONE freezes translation at the first selected source frame. Body and
    root rotations are smoothed; finger landmarks are only cropped.
    """
    if not isinstance(motion, MotionData):
        raise ValueError("motion must be a MotionData instance")
    body = np.asarray(motion.body_pose)
    if body.ndim != 3 or body.shape[1:] != (21, 3) or len(body) == 0:
        raise ValueError("body_pose must have non-empty shape [F,21,3]")
    frame_count = len(body)
    for name in ("source_start", "source_end"):
        value = source_start if name == "source_start" else source_end
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    end = frame_count if source_end == 0 else int(source_end)
    if not 1 <= source_start <= end <= frame_count:
        raise ValueError(f"Source frame range must satisfy 1 <= start <= end <= {frame_count}")
    speed = _number(speed, "speed", 0.1, 4.0)
    fps = _number(motion.fps, "fps", 1.0, 240.0)
    rotation_smoothing = _number(rotation_smoothing, "rotation_smoothing", 0.0, 0.5)
    translation_smoothing = _number(translation_smoothing, "translation_smoothing", 0.0, 0.5)
    if root_motion not in ("FULL", "IN_PLACE", "NONE"):
        raise ValueError("root_motion must be FULL, IN_PLACE, or NONE")

    selected = slice(int(source_start) - 1, end)
    updates = {}
    for field in fields(motion):
        value = getattr(motion, field.name)
        if field.name in _FRAME_FIELDS and value is not None:
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != frame_count:
                raise ValueError(f"{field.name} frame count must match body_pose")
            if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
                raise ValueError(f"{field.name} must contain finite numeric values")
            updates[field.name] = array[selected].astype(np.float64, copy=True)
        elif isinstance(value, np.ndarray):
            updates[field.name] = value.copy()
    for name in ("global_orient", "transl"):
        if updates.get(name) is None or updates[name].shape != (end - source_start + 1, 3):
            raise ValueError(f"{name} must have shape [F,3]")

    count = len(updates["body_pose"])
    radius = _radius(rotation_smoothing, fps, count)
    rotations = np.concatenate((updates["global_orient"][:, None, :], updates["body_pose"]), axis=1)
    rotations = _smooth_rotations(rotations, radius)
    updates["global_orient"] = rotations[:, 0].copy()
    updates["body_pose"] = rotations[:, 1:].copy()
    original_origin = updates["transl"][0].copy()
    translation = _smooth_values(updates["transl"], _radius(translation_smoothing, fps, count))
    if root_motion == "IN_PLACE":
        translation[:, (0, 2)] = original_origin[[0, 2]]
    elif root_motion == "NONE":
        translation[:] = original_origin
    updates["transl"] = translation
    updates["fps"] = fps * speed
    return replace(motion, **updates)
