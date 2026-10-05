"""Synthetic regression coverage for motion preparation (no Blender required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender_addon"))

from gvhmr_mmd_bridge.core import MotionData, axis_angle_to_matrix
from gvhmr_mmd_bridge.processing import prepare_motion


def make_motion(frames: int = 31, fps: float = 30.0) -> MotionData:
    timeline = np.arange(frames, dtype=np.float64)
    landmarks = np.broadcast_to(timeline[:, None, None], (frames, 21, 3)).copy()
    return MotionData(
        body_pose=np.zeros((frames, 21, 3)),
        global_orient=np.zeros((frames, 3)),
        transl=np.column_stack((timeline, timeline * 2, timeline * 3)),
        fps=fps,
        betas=np.arange(10, dtype=np.float64),
        static_confidence=np.column_stack((timeline, timeline + 1)),
        source_name="example.mp4",
        left_hand_landmarks=landmarks,
        right_hand_landmarks=landmarks + 100,
        left_hand_confidence=timeline / max(frames, 1),
        right_hand_confidence=timeline / max(frames, 1),
        hand_backend="synthetic",
    )


class PrepareMotionTests(unittest.TestCase):
    def test_default_preserves_values_without_aliasing(self):
        motion = make_motion()
        result = prepare_motion(motion)
        for name, value in vars(motion).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(result, name), value)
                self.assertFalse(np.shares_memory(getattr(result, name), value), name)
            else:
                self.assertEqual(getattr(result, name), value)

    def test_crop_is_inclusive_and_keeps_all_channels_synchronized(self):
        motion = make_motion()
        result = prepare_motion(motion, source_start=3, source_end=7, speed=2)
        self.assertEqual(result.frame_count, 5)
        self.assertEqual(result.fps, 60)
        for name in (
            "body_pose", "global_orient", "transl", "static_confidence",
            "left_hand_landmarks", "right_hand_landmarks",
            "left_hand_confidence", "right_hand_confidence",
        ):
            np.testing.assert_array_equal(getattr(result, name), getattr(motion, name)[2:7])
        np.testing.assert_array_equal(result.betas, motion.betas)
        self.assertEqual(result.source_name, motion.source_name)
        self.assertEqual(result.hand_backend, motion.hand_backend)
        self.assertEqual(prepare_motion(motion, source_start=30).frame_count, 2)

    def test_numpy_integer_range_is_supported(self):
        result = prepare_motion(make_motion(), source_start=np.int64(2), source_end=np.int32(3))
        self.assertEqual(result.frame_count, 2)

    def test_speed_changes_duration_without_discarding_samples(self):
        motion = make_motion()
        result = prepare_motion(motion, speed=0.5)
        self.assertEqual(result.fps, 15)
        self.assertEqual(result.frame_count, motion.frame_count)
        np.testing.assert_array_equal(result.body_pose, motion.body_pose)

    def test_single_frame_remains_finite(self):
        motion = make_motion(1)
        motion.body_pose[:] = [0.1, 0.2, -0.3]
        for mode in ("FULL", "IN_PLACE", "NONE"):
            result = prepare_motion(
                motion, rotation_smoothing=0.5, translation_smoothing=0.5, root_motion=mode
            )
            self.assertEqual(result.frame_count, 1)
            np.testing.assert_array_equal(result.body_pose, motion.body_pose)
            self.assertTrue(np.isfinite(result.transl).all())

    def test_constant_pose_and_translation_are_unchanged(self):
        motion = make_motion()
        motion.body_pose[:] = [0.2, -0.4, 0.5]
        motion.global_orient[:] = [0.5, 0.2, -0.1]
        motion.transl[:] = [3, 4, 5]
        result = prepare_motion(motion, rotation_smoothing=0.5, translation_smoothing=0.5)
        np.testing.assert_allclose(result.body_pose, motion.body_pose, atol=1e-12)
        np.testing.assert_allclose(result.global_orient, motion.global_orient, atol=1e-12)
        np.testing.assert_allclose(result.transl, motion.transl, atol=1e-12)

    def test_rotation_smoothing_takes_short_path_across_pi(self):
        motion = make_motion(21)
        angle = np.where(np.arange(21) % 2 == 0, np.pi - 0.05, -np.pi + 0.05)
        motion.global_orient[:, 1] = angle
        motion.body_pose[:, 3, 2] = angle
        result = prepare_motion(motion, rotation_smoothing=0.2)
        # Both source orientations are near a half turn, never near identity.
        self.assertTrue(np.all(np.abs(result.global_orient[:, 1]) > 3.0))
        self.assertTrue(np.all(np.abs(result.body_pose[:, 3, 2]) > 3.0))
        before = axis_angle_to_matrix(motion.global_orient)
        after = axis_angle_to_matrix(result.global_orient)
        self.assertLess(np.linalg.norm(np.diff(after, axis=0)),
                        np.linalg.norm(np.diff(before, axis=0)) * 0.35)
        self.assertTrue(np.isfinite(result.body_pose).all())

    def test_quaternion_sign_equivalents_do_not_cancel(self):
        motion = make_motion(15)
        motion.global_orient[:, 2] = np.where(np.arange(15) % 2, 0.3 + 2 * np.pi, 0.3)
        result = prepare_motion(motion, rotation_smoothing=0.3)
        np.testing.assert_allclose(
            axis_angle_to_matrix(result.global_orient),
            axis_angle_to_matrix(motion.global_orient), atol=1e-12,
        )

    def test_smoothing_reduces_jitter_and_preserves_source(self):
        motion = make_motion(121, fps=60)
        rng = np.random.default_rng(4)
        motion.body_pose[:] = rng.normal(0.0, 0.04, motion.body_pose.shape)
        motion.transl[:] = rng.normal(0.0, 0.03, motion.transl.shape)
        body_before, transl_before = motion.body_pose.copy(), motion.transl.copy()
        result = prepare_motion(motion, rotation_smoothing=0.15, translation_smoothing=0.15)
        self.assertLess(np.std(result.body_pose), np.std(motion.body_pose) * 0.5)
        self.assertLess(np.std(result.transl), np.std(motion.transl) * 0.5)
        self.assertTrue(np.isfinite(result.body_pose).all())
        np.testing.assert_array_equal(motion.body_pose, body_before)
        np.testing.assert_array_equal(motion.transl, transl_before)

    def test_centered_window_has_no_phase_shift(self):
        motion = make_motion(41)
        motion.transl[:] = 0
        motion.transl[20, 0] = 1
        result = prepare_motion(motion, translation_smoothing=0.2)
        self.assertEqual(int(np.argmax(result.transl[:, 0])), 20)
        np.testing.assert_array_equal(result.transl[:, 0], result.transl[::-1, 0])

    def test_root_modes_preserve_height_or_freeze_all_translation(self):
        motion = make_motion()
        full = prepare_motion(motion, source_start=3, source_end=7)
        np.testing.assert_array_equal(full.transl, motion.transl[2:7])
        stationary = prepare_motion(motion, source_start=3, source_end=7, root_motion="IN_PLACE")
        np.testing.assert_array_equal(stationary.transl[:, 1], full.transl[:, 1])
        np.testing.assert_array_equal(stationary.transl[:, 0], np.full(5, motion.transl[2, 0]))
        np.testing.assert_array_equal(stationary.transl[:, 2], np.full(5, motion.transl[2, 2]))
        frozen = prepare_motion(motion, source_start=3, source_end=7,
                                root_motion="NONE", translation_smoothing=0.5)
        np.testing.assert_array_equal(frozen.transl, np.broadcast_to(motion.transl[2], (5, 3)))

    def test_invalid_options_are_rejected(self):
        cases = (
            {"source_start": 0}, {"source_start": 32}, {"source_start": 2.0},
            {"source_start": True}, {"source_end": -1}, {"source_end": 32},
            {"source_start": 5, "source_end": 4}, {"source_end": False},
            {"speed": 0}, {"speed": 4.01}, {"speed": 0.09}, {"speed": np.nan},
            {"speed": np.inf}, {"speed": "2"}, {"speed": True},
            {"rotation_smoothing": -0.1}, {"rotation_smoothing": 0.51},
            {"translation_smoothing": np.nan}, {"translation_smoothing": np.inf},
            {"root_motion": "invalid"},
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_motion(make_motion(), **options)

    def test_invalid_motion_data_is_rejected(self):
        from dataclasses import replace

        motion = make_motion()
        cases = (
            replace(motion, body_pose=np.zeros((0, 21, 3))),
            replace(motion, global_orient=np.zeros((31, 1, 3))),
            replace(motion, transl=np.zeros((30, 3))),
            replace(motion, transl=np.full((31, 3), np.nan)),
            replace(motion, left_hand_landmarks=np.zeros((30, 21, 3))),
            replace(motion, static_confidence=np.asarray(1.0)),
            replace(motion, fps=np.inf),
        )
        for bad in cases:
            with self.subTest(data=bad), self.assertRaises(ValueError):
                prepare_motion(bad)

    def test_shorter_than_one_frame_window_is_a_no_op(self):
        motion = make_motion(5)
        motion.body_pose[2] = 0.1
        result = prepare_motion(motion, rotation_smoothing=0.001, translation_smoothing=0.001)
        np.testing.assert_array_equal(result.body_pose, motion.body_pose)
        np.testing.assert_array_equal(result.transl, motion.transl)


if __name__ == "__main__":
    unittest.main()
