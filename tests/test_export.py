"""GVHMR export validation without importing Blender or installing PyTorch."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from server.gvhmr_export import FORMAT_VERSION, build_payload, convert_pt_to_npz


def prediction(frame_count=3):
    return {
        "smpl_params_global": {
            "body_pose": np.arange(frame_count * 63, dtype=np.float32).reshape(frame_count, 21, 3),
            "global_orient": np.zeros((frame_count, 3)),
            "transl": np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3),
        }
    }


class ExportTests(unittest.TestCase):
    def test_single_frame_joint_layout_keeps_frame_axis(self):
        pred = prediction(1)
        payload = build_payload(pred)
        self.assertEqual(payload["body_pose"].shape, (1, 21, 3))
        np.testing.assert_array_equal(
            payload["body_pose"], pred["smpl_params_global"]["body_pose"]
        )

    def test_supported_body_layouts_preserve_frames_and_values(self):
        for frames in (1, 4):
            expected = prediction(frames)["smpl_params_global"]["body_pose"]
            layouts = (
                expected,
                expected.reshape(frames, 63),
                expected[None],
                expected.reshape(1, frames, 63),
            )
            for body in layouts:
                with self.subTest(frames=frames, shape=body.shape):
                    pred = prediction(frames)
                    pred["smpl_params_global"]["body_pose"] = body
                    payload = build_payload(pred)
                    np.testing.assert_array_equal(payload["body_pose"], expected)
                    self.assertEqual(payload["body_pose"].dtype, np.float32)

    def test_singleton_batch_is_removed_from_orient_and_translation(self):
        pred = prediction(4)
        params = pred["smpl_params_global"]
        expected_translation = params["transl"].copy()
        params["body_pose"] = params["body_pose"][None]
        params["global_orient"] = params["global_orient"][None]
        params["transl"] = params["transl"][None]
        payload = build_payload(pred)
        self.assertEqual(payload["global_orient"].shape, (4, 3))
        np.testing.assert_array_equal(payload["transl"], expected_translation)

    def test_metadata_and_optional_arrays(self):
        pred = prediction(3)
        pred["smpl_params_global"]["betas"] = np.arange(30).reshape(1, 3, 10)
        pred["net_outputs"] = {"static_conf_logits": np.ones((1, 3, 4))}
        pred["K_fullimg"] = np.eye(3)
        payload = build_payload(pred, fps=59.94, source_name="dance.pt")
        self.assertEqual(int(payload["format_version"]), FORMAT_VERSION)
        self.assertEqual(str(payload["source_name"]), "dance.pt")
        self.assertEqual(str(payload["coordinate_system"]), "smpl_y_up")
        self.assertAlmostEqual(float(payload["fps"]), 59.94, places=4)
        np.testing.assert_array_equal(payload["betas"], np.arange(10) + 10)
        self.assertEqual(payload["static_confidence"].shape, (3, 4))
        np.testing.assert_array_equal(payload["K_fullimg"], np.eye(3))

    def test_mismatched_optional_confidence_is_omitted(self):
        pred = prediction(3)
        pred["net_outputs"] = {"static_conf_logits": np.ones((2, 4))}
        self.assertNotIn("static_confidence", build_payload(pred))

    def test_missing_required_parameters_fail_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "smpl_params_global"):
            build_payload({})
        for name in ("body_pose", "global_orient", "transl"):
            with self.subTest(name=name):
                pred = prediction()
                del pred["smpl_params_global"][name]
                with self.assertRaisesRegex(ValueError, name):
                    build_payload(pred)

    def test_malformed_body_layout_is_rejected(self):
        for shape in ((3, 20, 3), (3, 62), (2, 3, 21, 3)):
            with self.subTest(shape=shape):
                pred = prediction()
                pred["smpl_params_global"]["body_pose"] = np.zeros(shape)
                with self.assertRaisesRegex(ValueError, "Unexpected body_pose shape"):
                    build_payload(pred)

    def test_empty_motion_and_frame_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no frames"):
            build_payload(prediction(0))
        for name in ("global_orient", "transl"):
            with self.subTest(name=name):
                pred = prediction()
                pred["smpl_params_global"][name] = np.zeros((2, 3))
                with self.assertRaisesRegex(ValueError, "Frame mismatch"):
                    build_payload(pred)

    def test_nonfinite_and_nonnumeric_motion_are_rejected(self):
        for name in ("body_pose", "global_orient", "transl"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(name=name, value=value):
                    pred = prediction()
                    pred["smpl_params_global"][name].flat[0] = value
                    with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                        build_payload(pred)
        pred = prediction()
        pred["smpl_params_global"]["body_pose"] = np.full((3, 21, 3), "invalid")
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            build_payload(pred)

    def test_invalid_fps_is_rejected(self):
        for fps in (0, -1, 0.5, 240.1, np.nan, np.inf):
            with self.subTest(fps=fps):
                with self.assertRaisesRegex(ValueError, "Invalid FPS"):
                    build_payload(prediction(), fps=fps)

    def test_convert_writes_pickle_free_npz_with_mocked_torch(self):
        pred = prediction(1)
        loader = Mock(return_value=pred)
        with tempfile.TemporaryDirectory() as directory:
            pt_path = Path(directory) / "hmr4d_results.pt"
            output_path = Path(directory) / "nested" / "motion.npz"
            with patch.dict(sys.modules, {"torch": SimpleNamespace(load=loader)}):
                result = convert_pt_to_npz(pt_path, output_path, fps=24)
            self.assertEqual(result, str(output_path))
            loader.assert_called_once_with(pt_path, map_location="cpu", weights_only=False)
            with np.load(output_path, allow_pickle=False) as stored:
                self.assertEqual(stored["body_pose"].shape, (1, 21, 3))
                self.assertEqual(float(stored["fps"]), 24)
                self.assertEqual(str(stored["source_name"]), "hmr4d_results.pt")
                for name in stored.files:
                    self.assertNotEqual(stored[name].dtype.kind, "O")


if __name__ == "__main__":
    unittest.main()
