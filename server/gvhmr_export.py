#!/usr/bin/env python3
"""Export an official GVHMR ``hmr4d_results.pt`` as a safe NumPy NPZ.

The resulting file contains numeric/string arrays only and can be loaded by
Blender with ``allow_pickle=False``.  PyTorch is needed only on the GVHMR
server, never inside Blender.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2


def _to_numpy(value, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def build_payload(pred: dict, *, fps: float = 30.0, source_name: str = "") -> dict:
    if "smpl_params_global" not in pred:
        raise ValueError("GVHMR result does not contain smpl_params_global")
    params = pred["smpl_params_global"]
    missing = sorted({"body_pose", "global_orient", "transl"}.difference(params))
    if missing:
        raise ValueError("Missing global SMPL parameters: " + ", ".join(missing))

    body_pose = _to_numpy(params["body_pose"], "body_pose")
    # Remove a singleton batch axis only from explicitly batched layouts.
    # A (1, 21, 3) array is already one frame, not a batch of 21 frames.
    if (
        body_pose.ndim == 4
        and body_pose.shape[0] == 1
        and body_pose.shape[-2:] == (21, 3)
    ) or (body_pose.ndim == 3 and body_pose.shape[0] == 1 and body_pose.shape[-1] == 63):
        body_pose = body_pose[0]
    if body_pose.ndim == 2 and body_pose.shape[1] == 63:
        body_pose = body_pose.reshape(-1, 21, 3)
    elif body_pose.ndim == 3 and body_pose.shape[1:] == (21, 3):
        pass
    else:
        raise ValueError(f"Unexpected body_pose shape: {body_pose.shape}")

    global_orient = _to_numpy(params["global_orient"], "global_orient")
    if global_orient.ndim == 3 and global_orient.shape[0] == 1:
        global_orient = global_orient[0]
    global_orient = global_orient.reshape(-1, 3)
    transl = _to_numpy(params["transl"], "transl")
    if transl.ndim == 3 and transl.shape[0] == 1:
        transl = transl[0]
    transl = transl.reshape(-1, 3)
    frame_count = body_pose.shape[0]
    if global_orient.shape[0] != frame_count or transl.shape[0] != frame_count:
        raise ValueError(
            f"Frame mismatch: body={frame_count}, orient={global_orient.shape[0]}, "
            f"transl={transl.shape[0]}"
        )
    if frame_count == 0:
        raise ValueError("GVHMR result contains no frames")
    if not np.isfinite(fps) or not 1.0 <= float(fps) <= 240.0:
        raise ValueError(f"Invalid FPS: {fps}")

    payload = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
        "body_pose": body_pose.astype(np.float32, copy=False),
        "global_orient": global_orient.astype(np.float32, copy=False),
        "transl": transl.astype(np.float32, copy=False),
        "fps": np.asarray(fps, dtype=np.float32),
        "coordinate_system": np.asarray("smpl_y_up"),
        "source_name": np.asarray(source_name),
    }

    if "betas" in params:
        betas = _to_numpy(params["betas"], "betas")
        while betas.ndim > 1:
            betas = betas.mean(axis=0)
        payload["betas"] = betas.reshape(-1).astype(np.float32, copy=False)

    net_outputs = pred.get("net_outputs")
    if isinstance(net_outputs, dict) and "static_conf_logits" in net_outputs:
        confidence = _to_numpy(net_outputs["static_conf_logits"], "static_confidence")
        if confidence.ndim == 3 and confidence.shape[0] == 1:
            confidence = confidence[0]
        if confidence.shape[0] == frame_count:
            payload["static_confidence"] = confidence

    if "K_fullimg" in pred:
        payload["K_fullimg"] = _to_numpy(pred["K_fullimg"], "K_fullimg")
    return payload


def convert_pt_to_npz(
    pt_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None,
    *,
    fps: float = 30.0,
) -> str:
    import torch

    pt_path = Path(pt_path)
    if output_path is None:
        output_path = pt_path.with_name("gvhmr_motion.npz")
    output_path = Path(output_path)

    pred = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(pred, dict):
        raise ValueError(f"Expected a dict in {pt_path}, got {type(pred).__name__}")
    payload = build_payload(pred, fps=fps, source_name=pt_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="GVHMR hmr4d_results.pt")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    output = convert_pt_to_npz(args.input, args.output, fps=args.fps)
    size_mb = Path(output).stat().st_size / (1024 * 1024)
    print(f"Exported {output} ({size_mb:.2f} MiB)")


if __name__ == "__main__":
    main()
