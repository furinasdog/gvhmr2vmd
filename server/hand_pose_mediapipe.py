#!/usr/bin/env python3
"""Add temporally smoothed MediaPipe 3D hand landmarks to a motion NPZ."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: Path):
        self.capture = cv2.VideoCapture(str(path))
        self.process = None
        if self.capture.isOpened():
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
            return
        self.capture.release()
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        self.width, self.height = int(stream["width"]), int(stream["height"])
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        self.fps = numerator / denominator if denominator else 30.0
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-i", str(path),
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
            ],
            stdout=subprocess.PIPE,
        )

    def read(self):
        if self.process is None:
            return self.capture.read()
        size = self.width * self.height * 3
        data = self.process.stdout.read(size)
        if len(data) != size:
            return False, None
        return True, np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)

    def close(self):
        if self.process is None:
            self.capture.release()
            return
        if self.process.stdout:
            self.process.stdout.close()
        self.process.terminate()
        self.process.wait(timeout=10)


def _canonicalize(landmarks) -> np.ndarray | None:
    points = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    x_axis = points[5] - points[17]  # little side -> index/thumb side
    y_axis = points[9] - points[0]   # wrist -> middle MCP
    x_norm, y_norm = np.linalg.norm(x_axis), np.linalg.norm(y_axis)
    if x_norm < 1e-6 or y_norm < 1e-6:
        return None
    x_axis /= x_norm
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return None
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    basis = np.stack((x_axis, y_axis, z_axis), axis=1)
    canonical = (points - points[0]) @ basis
    scale = max(float(np.linalg.norm(canonical[9])), 1e-6)
    return canonical / scale


def _fill_short_gaps(values: np.ndarray, confidence: np.ndarray, max_gap: int = 12) -> None:
    valid = np.flatnonzero(confidence > 0)
    for left, right in zip(valid[:-1], valid[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap:
            continue
        for offset in range(1, gap + 1):
            weight = offset / (gap + 1)
            values[left + offset] = (1.0 - weight) * values[left] + weight * values[right]
            confidence[left + offset] = 0.5 * min(confidence[left], confidence[right])


def _smooth(values: np.ndarray, confidence: np.ndarray, alpha: float = 0.45) -> None:
    previous = None
    for index in range(len(values)):
        if confidence[index] <= 0:
            previous = None
            continue
        if previous is not None:
            values[index] = alpha * values[index] + (1.0 - alpha) * previous
        previous = values[index].copy()


def extract_hands(video_path: Path, vitpose_path: Path, model_path: Path):
    import mediapipe as mp
    import torch

    keypoints = torch.load(vitpose_path, map_location="cpu", weights_only=False)
    keypoints = np.asarray(keypoints.cpu() if hasattr(keypoints, "cpu") else keypoints)
    if keypoints.ndim != 3 or keypoints.shape[1] < 11:
        raise ValueError(f"Unexpected ViTPose shape: {keypoints.shape}")

    reader = VideoReader(video_path)
    fps = reader.fps
    frame_count = len(keypoints)
    results = {
        side: {
            "landmarks": np.zeros((frame_count, 21, 3), dtype=np.float32),
            "confidence": np.zeros(frame_count, dtype=np.float32),
        }
        for side in ("left", "right")
    }

    BaseOptions = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.20,
        min_hand_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )
    landmarkers = {
        side: vision.HandLandmarker.create_from_options(options)
        for side in ("left", "right")
    }
    previous_crop = {"left": None, "right": None}
    try:
        for frame_index in range(frame_count):
            ok, frame = reader.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            for side, wrist_index, elbow_index in (
                ("left", 9, 7),
                ("right", 10, 8),
            ):
                wrist = keypoints[frame_index, wrist_index, :2].astype(float)
                elbow = keypoints[frame_index, elbow_index, :2].astype(float)
                score = min(
                    float(keypoints[frame_index, wrist_index, 2]),
                    float(keypoints[frame_index, elbow_index, 2]),
                )
                if score >= 0.15:
                    forearm = max(float(np.linalg.norm(wrist - elbow)), 24.0)
                    center = wrist + 0.32 * (wrist - elbow)
                    crop_size = int(np.clip(2.2 * forearm, 96, min(width, height)))
                    previous_crop[side] = (center, crop_size)
                crop_info = previous_crop[side]
                if crop_info is None:
                    continue
                center, crop_size = crop_info
                crop = cv2.getRectSubPix(
                    frame,
                    (int(crop_size), int(crop_size)),
                    (float(center[0]), float(center[1])),
                )
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detected = landmarkers[side].detect_for_video(image, timestamp_ms)
                if not detected.hand_world_landmarks:
                    continue
                canonical = _canonicalize(detected.hand_world_landmarks[0])
                if canonical is None:
                    continue
                confidence = 1.0
                if detected.handedness and detected.handedness[0]:
                    confidence = float(detected.handedness[0][0].score)
                results[side]["landmarks"][frame_index] = canonical
                results[side]["confidence"][frame_index] = np.clip(confidence, 0.0, 1.0)
    finally:
        reader.close()
        for landmarker in landmarkers.values():
            landmarker.close()

    for side in ("left", "right"):
        _fill_short_gaps(results[side]["landmarks"], results[side]["confidence"])
        _smooth(results[side]["landmarks"], results[side]["confidence"])
    return results, frame_count


def merge_into_motion_npz(motion_path: Path, results: dict, frame_count: int) -> None:
    with np.load(motion_path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    body_frames = int(payload["body_pose"].shape[0])
    if frame_count != body_frames:
        raise ValueError(f"Hand/body frame mismatch: hand={frame_count}, body={body_frames}")
    payload.update(
        {
            "format_version": np.asarray(2, dtype=np.int32),
            "left_hand_landmarks": results["left"]["landmarks"],
            "right_hand_landmarks": results["right"]["landmarks"],
            "left_hand_confidence": results["left"]["confidence"],
            "right_hand_confidence": results["right"]["confidence"],
            "hand_backend": np.asarray("mediapipe-hand-landmarker"),
            "hand_coordinate_system": np.asarray("canonical_palm_v1"),
        }
    )
    temporary = motion_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, motion_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--vitpose", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--motion-npz", type=Path, required=True)
    args = parser.parse_args()
    results, frame_count = extract_hands(args.video, args.vitpose, args.model)
    merge_into_motion_npz(args.motion_npz, results, frame_count)
    rates = {
        side: 100.0 * float(np.count_nonzero(results[side]["confidence"])) / frame_count
        for side in ("left", "right")
    }
    print(
        f"HAND_POSE_OK frames={frame_count} "
        f"left={rates['left']:.1f}% right={rates['right']:.1f}%"
    )


if __name__ == "__main__":
    main()
