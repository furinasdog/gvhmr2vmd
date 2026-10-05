"""Blender-independent regression tests for the public motion archive contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender_addon"))

from gvhmr_mmd_bridge.core import load_motion  # noqa: E402


class LoadMotionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "motion.npz"

    @staticmethod
    def payload(frames=2):
        return {
            "format_version": np.asarray(2, dtype=np.int32),
            "body_pose": np.arange(frames * 63, dtype=np.float32).reshape(frames, 21, 3) / 100,
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "transl": np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
            "fps": np.asarray(30, dtype=np.float32),
        }

    def load(self, payload):
        np.savez_compressed(self.path, **payload)
        return load_motion(self.path)

    def assert_invalid(self, field, value, message=None):
        payload = self.payload()
        payload[field] = value
        with self.assertRaisesRegex(ValueError, message or field):
            self.load(payload)

    def test_single_frame_is_not_mistaken_for_a_batch(self):
        payload = self.payload(frames=1)
        motion = self.load(payload)
        self.assertEqual(motion.frame_count, 1)
        np.testing.assert_array_equal(motion.body_pose, payload["body_pose"])
        self.assertEqual(motion.global_orient.shape, (1, 3))
        self.assertEqual(motion.transl.shape, (1, 3))

    def test_supported_body_shapes_preserve_frames_and_values(self):
        for frames in (1, 4):
            payload = self.payload(frames=frames)
            canonical = payload["body_pose"]
            for pose in (
                canonical,
                canonical.reshape(frames, 63),
                canonical[None],
                canonical.reshape(1, frames, 63),
            ):
                with self.subTest(frames=frames, shape=pose.shape):
                    payload["body_pose"] = pose
                    motion = self.load(payload)
                    self.assertEqual(motion.frame_count, frames)
                    np.testing.assert_array_equal(motion.body_pose, canonical)

    def test_legacy_singleton_batches_are_supported(self):
        payload = self.payload()
        for key in ("body_pose", "global_orient", "transl"):
            payload[key] = payload[key][None]
        payload["static_confidence"] = np.asarray([[[-3, 4], [5, -6]]])
        motion = self.load(payload)
        self.assertEqual(motion.body_pose.shape, (2, 21, 3))
        self.assertEqual(motion.global_orient.shape, (2, 3))
        self.assertEqual(motion.transl.shape, (2, 3))
        # The exporter supplies logits here, so these values are not probabilities.
        np.testing.assert_array_equal(motion.static_confidence, [[-3, 4], [5, -6]])

    def test_v1_and_v2_body_only_archives_are_supported(self):
        for version in (1, 2):
            with self.subTest(version=version):
                payload = self.payload()
                payload["format_version"] = np.asarray(version)
                motion = self.load(payload)
                self.assertEqual(motion.fps, 30)
                self.assertIsNone(motion.left_hand_landmarks)
                self.assertIsNone(motion.right_hand_landmarks)
                self.assertEqual(motion.source_name, "")

    def test_single_value_metadata_arrays_are_supported(self):
        payload = self.payload()
        payload["format_version"] = np.asarray([2])
        payload["fps"] = np.asarray([29.97])
        payload["source_name"] = np.asarray(["dance.mp4"])
        payload["hand_backend"] = np.asarray("mediapipe-hand-landmarker")
        motion = self.load(payload)
        self.assertAlmostEqual(motion.fps, 29.97)
        self.assertEqual(motion.source_name, "dance.mp4")
        self.assertEqual(motion.hand_backend, "mediapipe-hand-landmarker")

    def test_utf8_byte_metadata_is_decoded(self):
        payload = self.payload()
        payload["source_name"] = np.asarray(b"dance.mp4")
        self.assertEqual(self.load(payload).source_name, "dance.mp4")

    def test_missing_required_fields_have_actionable_errors(self):
        for key in self.payload():
            with self.subTest(field=key):
                payload = self.payload()
                del payload[key]
                with self.assertRaisesRegex(ValueError, f"Missing NPZ fields: {key}"):
                    self.load(payload)

    def test_scalar_metadata_rejects_empty_and_multiple_values(self):
        for key in ("format_version", "fps"):
            for value in (np.asarray([]), np.asarray([1, 2])):
                with self.subTest(field=key, shape=value.shape):
                    self.assert_invalid(key, value, f"{key} must contain exactly one")

    def test_scalar_metadata_rejects_non_numeric_and_non_finite_values(self):
        for key in ("format_version", "fps"):
            for value in ("2", True, complex(2), np.nan, np.inf, -np.inf):
                with self.subTest(field=key, value=value):
                    self.assert_invalid(key, np.asarray(value))

    def test_fractional_version_is_not_truncated(self):
        self.assert_invalid("format_version", np.asarray(1.5), "must be an integer")

    def test_unsupported_format_version_is_rejected(self):
        for version in (0, 3, -1):
            with self.subTest(version=version):
                self.assert_invalid("format_version", np.asarray(version), "Unsupported motion")

    def test_fps_range_is_enforced_with_inclusive_limits(self):
        for fps in (1, 240):
            payload = self.payload()
            payload["fps"] = np.asarray(fps)
            self.assertEqual(self.load(payload).fps, fps)
        for fps in (0, 0.99, 240.01, -30):
            self.assert_invalid("fps", np.asarray(fps), "Invalid FPS")

    def test_body_pose_rejects_incorrect_shapes(self):
        for shape in ((), (126,), (21, 3), (2, 3, 21), (2, 1, 21, 3), (1, 2, 1, 63)):
            with self.subTest(shape=shape):
                self.assert_invalid("body_pose", np.zeros(shape), "body_pose must have shape")

    def test_root_vectors_do_not_silently_reshape_incorrect_axes(self):
        for key in ("global_orient", "transl"):
            for shape in ((), (6,), (3, 2), (1, 6), (2, 1, 3), (1, 1, 6), (1, 1, 2, 3)):
                with self.subTest(field=key, shape=shape):
                    self.assert_invalid(key, np.zeros(shape), f"{key} must have shape")

    def test_mismatched_frame_counts_are_rejected(self):
        for key in ("global_orient", "transl"):
            with self.subTest(field=key):
                self.assert_invalid(key, np.zeros((1, 3)), "Frame count mismatch")

    def test_empty_motion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Motion contains no frames"):
            self.load(self.payload(frames=0))

    def test_motion_arrays_reject_non_numeric_and_non_finite_values(self):
        for key in ("body_pose", "global_orient", "transl"):
            for invalid in ("zero", np.nan, np.inf):
                with self.subTest(field=key, invalid=invalid):
                    value = np.full(self.payload()[key].shape, invalid)
                    self.assert_invalid(key, value)

    def test_string_metadata_rejects_empty_multiple_and_non_string_values(self):
        for key in ("source_name", "hand_backend"):
            for value in (np.asarray([], dtype="U"), np.asarray(["a", "b"]), np.asarray(12)):
                with self.subTest(field=key, value=value):
                    self.assert_invalid(key, value, f"{key} must contain exactly one string")

    def test_static_confidence_validates_shape_and_frame_count(self):
        for value in (np.asarray(1), np.asarray([]), np.empty((2, 0)), np.zeros((2, 1, 1))):
            with self.subTest(shape=value.shape):
                self.assert_invalid("static_confidence", value, "static_confidence must have shape")
        self.assert_invalid("static_confidence", np.zeros(1), "frame count does not match")
        payload = self.payload()
        payload["static_confidence"] = np.asarray([-2, 4])
        np.testing.assert_array_equal(self.load(payload).static_confidence, [-2, 4])

    def test_betas_reject_empty_and_non_finite_values(self):
        self.assert_invalid("betas", np.asarray([]), "betas must not be empty")
        self.assert_invalid("betas", np.asarray([np.nan]))
        payload = self.payload()
        payload["betas"] = np.zeros(10)
        self.assertEqual(self.load(payload).betas.shape, (10,))

    def test_hand_fields_round_trip_for_both_sides(self):
        payload = self.payload()
        for side in ("left", "right"):
            payload[f"{side}_hand_landmarks"] = np.zeros((2, 21, 3))
            payload[f"{side}_hand_confidence"] = np.asarray([0.0, 1.0])
        motion = self.load(payload)
        for side in ("left", "right"):
            np.testing.assert_array_equal(
                getattr(motion, f"{side}_hand_confidence"), [0.0, 1.0]
            )
            self.assertEqual(getattr(motion, f"{side}_hand_landmarks").shape, (2, 21, 3))

    def test_hand_landmarks_and_confidence_must_be_paired(self):
        for side in ("left", "right"):
            for field, value, missing in (
                ("landmarks", np.zeros((2, 21, 3)), "confidence"),
                ("confidence", np.ones(2), "landmarks"),
            ):
                with self.subTest(side=side, field=field):
                    self.assert_invalid(
                        f"{side}_hand_{field}", value, f"Missing NPZ field: {side}_hand_{missing}"
                    )

    def test_hand_confidence_rejects_invalid_shapes_and_values(self):
        for side in ("left", "right"):
            for value in (np.ones((2, 1)), np.ones((1, 2)), np.ones(1), np.ones(3),
                          np.asarray([-0.1, 0.5]), np.asarray([0.5, 1.1]),
                          np.asarray([np.nan, 1])):
                with self.subTest(side=side, value=value):
                    payload = self.payload()
                    payload[f"{side}_hand_landmarks"] = np.zeros((2, 21, 3))
                    payload[f"{side}_hand_confidence"] = value
                    with self.assertRaisesRegex(ValueError, f"{side}_hand_confidence"):
                        self.load(payload)

    def test_hand_landmarks_validate_frames_joints_and_coordinates(self):
        for side in ("left", "right"):
            for shape in ((1, 21, 3), (2, 20, 3), (2, 3, 21), (2, 63)):
                with self.subTest(side=side, shape=shape):
                    self.assert_invalid(
                        f"{side}_hand_landmarks", np.zeros(shape),
                        f"{side}_hand_landmarks must have shape",
                    )

    def test_object_arrays_are_never_unpickled(self):
        self.assert_invalid("body_pose", np.asarray([object()]), "allow_pickle=False")

    def test_wrong_extension_and_missing_file_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "npz extension"):
            load_motion(self.path.with_suffix(".pt"))
        with self.assertRaisesRegex(ValueError, "Motion file not found"):
            load_motion(self.path)

    def test_npy_file_disguised_as_npz_is_rejected(self):
        with self.path.open("wb") as output:
            np.save(output, np.zeros(3))
        with self.assertRaisesRegex(ValueError, "NumPy NPZ archive"):
            load_motion(self.path)


if __name__ == "__main__":
    unittest.main()
