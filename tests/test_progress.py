"""Real-log parsing and WebUI flow tests without GPU, Gradio, or PyTorch."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from server.progress import GVHMRProgressParser, StageUpdate, iter_log_records


class ProgressParserTests(unittest.TestCase):
    def test_official_tqdm_stages_preserve_reported_counts(self):
        parser = GVHMRProgressParser()
        for marker, stage in (
            ("Copy", "copy"),
            ("YoloV8 Tracking", "tracking"),
            ("ViTPose", "pose"),
            ("HMR2 Feature", "features"),
            ("DPVO", "camera"),
            ("Rendering Incam", "render_incam"),
            ("Rendering Global", "render_global"),
        ):
            with self.subTest(marker=marker):
                update = parser.parse(f"{marker}: 50%|#####     | 4/8 [00:01<00:01]")
                self.assertEqual((update.stage, update.completed, update.total), (stage, 4, 8))

    def test_split_utf8_ansi_cr_and_final_record_preserve_raw_log(self):
        raw = "\x1b[32m[Preprocess] Start!\x1b[0m\r\nViTPose: 50%|##| 4/8 [00:01]\r结束".encode()

        class ChunkedStream:
            chunks = iter([raw[index : index + 1] for index in range(len(raw))])

            def read(self, _size):
                return next(self.chunks, b"")

        saved = io.BytesIO()
        records = list(iter_log_records(ChunkedStream(), saved))
        self.assertEqual(records, ["[Preprocess] Start!", "ViTPose: 50%|##| 4/8 [00:01]", "结束"])
        self.assertEqual(saved.getvalue(), raw)

    def test_cached_stages_remain_explicit_and_do_not_invent_counts(self):
        parser = GVHMRProgressParser()
        for line in (
            "[Preprocess] bbx (xyxy, xys) from cached/bbx.pt",
            "[Preprocess] vitpose from cached/vitpose.pt",
            "[Preprocess] vit_features from cached/features.pt",
            "[Preprocess] slam results from cached/slam.pt",
            "[Render Incam] Video already exists at cached/1_incam.mp4",
            "[Render Global] Video already exists at cached/2_global.mp4",
        ):
            with self.subTest(line=line):
                update = parser.parse(line)
                self.assertIn("缓存", update.description)
                self.assertIsNone(update.total)
        self.assertFalse(parser.inference_started)

    def test_inference_and_unknown_output_never_get_a_guessed_percentage(self):
        parser = GVHMRProgressParser()
        update = parser.parse("[2026-10-05] [HMR4D] Predicting")
        self.assertEqual(update.stage, "inference")
        self.assertIsNone(update.total)
        self.assertTrue(parser.inference_started)
        self.assertIsNone(parser.parse("Loading checkpoint 54/100 MB"))
        self.assertIsNone(parser.parse("unexpected new upstream output"))
        self.assertEqual(parser.parse("[HMR4D] Elapsed: 3.14s").stage, "inference_save")

    def test_simple_vo_counter_requires_its_explicit_context(self):
        parser = GVHMRProgressParser()
        self.assertIsNone(parser.parse("50%|###| 2/4 [00:01]"))
        parser.parse("[SimpleVO] Choosen frames shape: (5, 960, 540, 3)")
        update = parser.parse("50%|###| 2/4 [00:01]")
        self.assertEqual((update.completed, update.total), (2, 4))
        self.assertIsNone(parser.parse("DPVO: 100%|###| 5/4 [00:01]").total)
        self.assertIsNone(parser.parse("ViTPose: 0it [00:00, ?it/s]").total)

    def test_gradio_calls_distinguish_stage_counts_from_unknown_total(self):
        progress = Mock()
        StageUpdate("pose", "姿态关键点", 4, 8).report(progress)
        self.assertEqual(progress.call_args.args[0], (4, 8))
        self.assertIn("仅本阶段", progress.call_args.kwargs["desc"])
        StageUpdate("inference", "GVHMR 推理").report(progress)
        self.assertEqual(progress.call_args.args[0], (0, None))
        StageUpdate("pose", "姿态关键点", 8, 8).report(progress)
        self.assertNotEqual(progress.call_args.args[0], 1.0)
        self.assertEqual(
            GVHMRProgressParser().parse("[Render] Skipped by --skip_render").stage, "render_skipped"
        )


class FakeProcess:
    def __init__(self, output, returncode, on_wait=None):
        self.stdout = io.BytesIO(output)
        self.returncode = returncode
        self.on_wait = on_wait

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.stdout.close()

    def wait(self):
        if self.on_wait:
            self.on_wait()
        return self.returncode


class WebUIProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "server" / "webui.py"
        spec = importlib.util.spec_from_file_location("server._progress_test_webui", path)
        cls.webui = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"gradio": SimpleNamespace(Progress=lambda: None)}):
            spec.loader.exec_module(cls.webui)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "input.mp4"
        self.video.write_bytes(b"video fixture")
        self.output = self.root / "outputs" / "input"
        self.output.mkdir(parents=True)
        self.progress = Mock()

    def run_flow(
        self,
        *,
        main_code=0,
        hands=False,
        hand_code=0,
        preview=False,
        export=True,
        export_error=False,
        write_pose=True,
        write_preview=True,
        cached=False,
        main_log=None,
    ):
        if cached:
            (self.output / "hmr4d_results.pt").write_bytes(b"old cached pose")
            (self.output / "gvhmr_motion.npz").write_bytes(b"old cached npz")
        if main_log is None:
            main_log = (
                b"[Preprocess] Start!\rYoloV8 Tracking: 50%|###| 1/2 [00:01]\r"
                b"ViTPose: 100%|###| 1/1 [00:01]\r[Preprocess] End.\n"
                b"[HMR4D] Predicting\n[HMR4D] Elapsed: 1s\n"
            )

        def write_outputs():
            if write_pose:
                (self.output / "hmr4d_results.pt").write_bytes(b"new pose")
            if preview and write_preview:
                (self.output / "1_incam.mp4").write_bytes(b"rendered video")

        processes = [FakeProcess(main_log, main_code, write_outputs)]
        if hands:
            processes.append(FakeProcess(b"hand details\nHAND_POSE_OK frames=2\n", hand_code))
        popen = Mock(side_effect=processes)

        def convert(_source, destination, **_kwargs):
            if export_error:
                raise ValueError("bad pose data")
            Path(destination).write_bytes(b"new npz")
            return str(destination)

        with ExitStack() as stack:
            for name, value in (
                ("PROJ_ROOT", self.root),
                ("OUTPUT_ROOT", self.root / "outputs"),
                ("INFERENCE_LOCK", threading.Lock()),
                ("check_models_ready", lambda: []),
                ("ensure_hand_model", lambda: None),
                ("convert_pt_to_npz", convert),
            ):
                stack.enter_context(patch.object(self.webui, name, value))
            stack.enter_context(patch.object(self.webui.subprocess, "Popen", popen))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = self.webui.run_gvhmr(
                self.video, True, False, 0, export, preview, hands, progress=self.progress
            )
            self.assertFalse(self.webui.INFERENCE_LOCK.locked())
        return result, popen

    def assert_not_success(self):
        self.assertFalse(any(call.args[0] == 1.0 for call in self.progress.call_args_list))

    def test_success_reports_real_stages_and_only_finishes_after_hands(self):
        result, popen = self.run_flow(hands=True, preview=True)
        self.assertEqual(popen.call_count, 2)
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")
        self.assertIsNotNone(result[1])
        self.assertEqual(self.progress.call_args.args[0], 1.0)
        descriptions = [call.kwargs["desc"] for call in self.progress.call_args_list]
        self.assertTrue(any("GVHMR 推理" in text for text in descriptions))
        self.assertTrue(any("手部识别" in text for text in descriptions))
        self.assertIn(b"hand details", (self.output / "webui_process.log").read_bytes())

    def test_disabled_steps_do_not_run_or_display_stale_outputs(self):
        result, popen = self.run_flow(export=False, cached=True)
        self.assertEqual(popen.call_count, 1)
        self.assertIn("--skip_render", popen.call_args.args[0])
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        self.assertFalse(
            any("手部识别" in call.kwargs["desc"] for call in self.progress.call_args_list)
        )
        self.assertEqual(self.progress.call_args.args[0], 1.0)

    def test_failed_process_with_old_cache_never_claims_success(self):
        result, _ = self.run_flow(main_code=1, cached=True, write_pose=False)
        self.assert_not_success()
        self.assertIn("可能来自已有缓存", result[3])
        self.assertIn("退出码 1", result[3])

    def test_successful_cache_reuse_is_identified(self):
        result, _ = self.run_flow(
            cached=True,
            write_pose=False,
            main_log=(
                b"[Preprocess] vitpose from cache/vitpose.pt\n[Preprocess] End.\n"
                b"[Render] Skipped by --skip_render\n"
            ),
        )
        self.assertEqual(self.progress.call_args.args[0], 1.0)
        self.assertIn("复用了已有姿态结果", result[3])

    def test_missing_output_is_failure_even_with_zero_exit(self):
        result, _ = self.run_flow(write_pose=False)
        self.assert_not_success()
        self.assertIsNone(result[0])
        self.assertIn("推理失败", result[3])

    def test_export_failure_does_not_return_previous_npz_or_run_hands(self):
        result, popen = self.run_flow(cached=True, export_error=True, hands=True)
        self.assert_not_success()
        self.assertIsNone(result[1])
        self.assertEqual(popen.call_count, 1)
        self.assertIn("NPZ 导出失败", result[3])

    def test_hand_failure_and_missing_preview_do_not_finish_task(self):
        result, _ = self.run_flow(hands=True, hand_code=1, preview=True, write_preview=False)
        self.assert_not_success()
        self.assertIn("手部识别失败", result[3])
        self.assertIn("未生成视频", result[3])

    def test_full_raw_logs_survive_tail_truncation(self):
        raw = b"".join(f"record {index}\r".encode() for index in range(100))
        result, _ = self.run_flow(main_log=raw, main_code=1, write_pose=False)
        self.assertEqual((self.output / "webui_process.log").read_bytes(), raw)
        self.assertIn("record 99", result[3])
        self.assert_not_success()

    def test_hands_without_npz_is_rejected_before_launch(self):
        result, popen = self.run_flow(hands=True, export=False)
        popen.assert_not_called()
        self.assertIn("需要启用 NPZ", result[3])
        self.assert_not_success()


if __name__ == "__main__":
    unittest.main()
