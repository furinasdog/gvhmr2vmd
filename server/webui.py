"""GVHMR WebUI: video in, preview plus Blender-safe motion NPZ out."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import gradio as gr

from gvhmr_export import convert_pt_to_npz


PROJ_ROOT = Path(__file__).parent.resolve()
CHECKPOINT_ROOT = PROJ_ROOT / "inputs" / "checkpoints"
OUTPUT_ROOT = PROJ_ROOT / "outputs" / "demo"
RESULTS_META = "webui_results.json"
HF_MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
INFERENCE_LOCK = threading.Lock()
HAND_MODEL = CHECKPOINT_ROOT / "mediapipe" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

MODELS = [
    # SMPL/SMPL-X require registration and acceptance of their own licenses.
    # They are intentionally never downloaded from unofficial mirrors here.
    ("body_models/smpl/SMPL_NEUTRAL.pkl", None, None, "body_models/smpl", "SMPL_NEUTRAL.pkl"),
    ("body_models/smplx/SMPLX_NEUTRAL.npz", None, None, "body_models/smplx", "SMPLX_NEUTRAL.npz"),
    ("dpvo/dpvo.pth", "camenduru/GVHMR", "dpvo/dpvo.pth", "dpvo", "dpvo.pth"),
    ("gvhmr/gvhmr_siga24_release.ckpt", "camenduru/GVHMR", "gvhmr/gvhmr_siga24_release.ckpt", "gvhmr", "gvhmr_siga24_release.ckpt"),
    ("hmr2/epoch=10-step=25000.ckpt", "camenduru/GVHMR", "hmr2/epoch%3D10-step%3D25000.ckpt", "hmr2", "epoch=10-step=25000.ckpt"),
    ("vitpose/vitpose-h-multi-coco.pth", "camenduru/GVHMR", "vitpose/vitpose-h-multi-coco.pth", "vitpose", "vitpose-h-multi-coco.pth"),
    ("yolo/yolov8x.pt", "camenduru/GVHMR", "yolo/yolov8x.pt", "yolo", "yolov8x.pt"),
]

MIN_SIZES_MB = {
    "SMPL_NEUTRAL.pkl": 100,
    "SMPLX_NEUTRAL.npz": 50,
    "dpvo.pth": 5,
    "gvhmr_siga24_release.ckpt": 50,
    "epoch=10-step=25000.ckpt": 1000,
    "vitpose-h-multi-coco.pth": 1000,
    "yolov8x.pt": 50,
}


def _is_valid_model(filepath: Path, filename: str) -> bool:
    return filepath.is_file() and filepath.stat().st_size >= MIN_SIZES_MB.get(filename, 1) * 1024 * 1024


def check_models_ready() -> list[str]:
    return [
        rel_path
        for rel_path, _repo, _remote_fn, _local_dir, local_fn in MODELS
        if not _is_valid_model(CHECKPOINT_ROOT / rel_path, local_fn)
    ]


def download_models_yield():
    missing = check_models_ready()
    if not missing:
        yield "All model weights are ready."
        return
    by_path = {entry[0]: entry for entry in MODELS}
    for index, rel_path in enumerate(missing, start=1):
        _rel, repo, remote_fn, local_dir, local_fn = by_path[rel_path]
        if repo is None:
            yield (
                f"[{index}/{len(missing)}] {local_fn} 需要从 SMPL/SMPL-X 官网"
                f"接受许可后手动放入 inputs/checkpoints/{local_dir}/。"
            )
            continue
        destination_dir = CHECKPOINT_ROOT / local_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / local_fn
        if destination.exists() and not _is_valid_model(destination, local_fn):
            destination.unlink()
        url = f"{HF_MIRROR}/{repo}/resolve/main/{remote_fn}"
        yield f"[{index}/{len(missing)}] Downloading {local_fn}..."
        result = subprocess.run(
            ["curl", "-L", "--fail", "-#", "-o", str(destination), url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not _is_valid_model(destination, local_fn):
            yield f"Failed to download {local_fn}: {result.stderr[-500:]}"
            return
        yield f"Finished {local_fn}."
        time.sleep(0.2)


def ensure_hand_model() -> None:
    if HAND_MODEL.is_file() and HAND_MODEL.stat().st_size > 5 * 1024 * 1024:
        return
    HAND_MODEL.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-L", "--fail", "-o", str(HAND_MODEL), HAND_MODEL_URL],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not HAND_MODEL.is_file():
        raise RuntimeError("手部模型下载失败：" + result.stderr[-500:])


def save_results_meta(output_dir: Path, files: dict) -> None:
    payload = {key: files.get(key) for key in ("pt", "npz", "video")}
    (output_dir / RESULTS_META).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_results_meta(output_dir: Path):
    meta_path = output_dir / RESULTS_META
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("pt", "npz", "video"):
        value = data.get(key)
        if value and not Path(value).is_file():
            data[key] = None
    return data


def find_latest_result():
    if not OUTPUT_ROOT.is_dir():
        return None
    candidates = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir() and (d / RESULTS_META).is_file()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda d: (d / RESULTS_META).stat().st_mtime)
    return load_results_meta(latest)


def collect_output_files(output_dir: Path) -> dict:
    pt_file = output_dir / "hmr4d_results.pt"
    npz_file = output_dir / "gvhmr_motion.npz"
    videos = sorted(output_dir.glob("*_3_incam_global_horiz.mp4"))
    render = videos[0] if videos else output_dir / "1_incam.mp4"
    if not render.is_file():
        render = output_dir / "2_global.mp4"
    return {
        "pt": str(pt_file) if pt_file.is_file() else None,
        "npz": str(npz_file) if npz_file.is_file() else None,
        "video": str(render) if render.is_file() else None,
    }


def run_gvhmr(
    video_path,
    static_cam,
    use_dpvo,
    focal_mm,
    export_npz,
    render_preview,
    recognize_hands,
    progress=gr.Progress(),
):
    if video_path is None:
        return None, None, None, "请先上传视频。"
    video_path = Path(video_path)
    if not video_path.is_file():
        return None, None, None, f"找不到视频：{video_path}"
    if not INFERENCE_LOCK.acquire(blocking=False):
        return None, None, None, "GPU 正在处理另一项任务，请稍后再试。"

    try:
        missing = check_models_ready()
        if missing:
            progress(0.0, desc="下载缺失模型…")
            download_log = []
            for line in download_models_yield():
                download_log.append(line)
                if line.startswith("Failed"):
                    return None, None, None, "\n".join(download_log)
            missing = check_models_ready()
            if missing:
                return None, None, None, (
                    "仍缺少需要手动提供的模型文件：\n"
                    + "\n".join(f"- {item}" for item in missing)
                    + "\n请按 GVHMR 官方安装文档取得相应许可和文件。"
                )

        command = [
            sys.executable,
            str(PROJ_ROOT / "tools" / "demo" / "demo.py"),
            f"--video={video_path}",
        ]
        if static_cam:
            command.append("--static_cam")
        if use_dpvo:
            command.append("--use_dpvo")
        if focal_mm and int(focal_mm) > 0:
            command.append(f"--f_mm={int(focal_mm)}")
        if not render_preview:
            command.append("--skip_render")

        progress(0.12, desc="GVHMR 推理中…")
        tail = deque(maxlen=40)
        process = subprocess.Popen(
            command,
            cwd=str(PROJ_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            tail.append(line.rstrip())
        return_code = process.wait()

        output_dir = OUTPUT_ROOT / video_path.stem
        files = collect_output_files(output_dir)
        if not files["pt"]:
            details = "\n".join(tail)
            return None, None, None, f"推理失败（退出码 {return_code}）。\n{details}"

        log = []
        if return_code != 0:
            log.append(f"渲染阶段返回 {return_code}，但姿态结果已生成。")
        if export_npz:
            progress(0.94, desc="导出 Blender NPZ…")
            try:
                files["npz"] = convert_pt_to_npz(
                    files["pt"],
                    output_dir / "gvhmr_motion.npz",
                    fps=30.0,
                )
                log.append("已导出 Blender 安全格式 NPZ。")
            except Exception as exc:
                log.append(f"NPZ 导出失败：{exc}")

        if recognize_hands and files["npz"]:
            progress(0.96, desc="识别双手与手指动作…")
            try:
                ensure_hand_model()
                normalized_video = output_dir / "0_input_video.mp4"
                hand_video = normalized_video if normalized_video.is_file() else video_path
                hand_command = [
                    sys.executable,
                    str(PROJ_ROOT / "hand_pose_mediapipe.py"),
                    "--video", str(hand_video),
                    "--vitpose", str(output_dir / "preprocess" / "vitpose.pt"),
                    "--model", str(HAND_MODEL),
                    "--motion-npz", str(files["npz"]),
                ]
                hand_process = subprocess.run(
                    hand_command,
                    cwd=str(PROJ_ROOT),
                    capture_output=True,
                    text=True,
                )
                if hand_process.returncode != 0:
                    raise RuntimeError((hand_process.stdout + hand_process.stderr)[-1500:])
                log.append(hand_process.stdout.strip())
            except Exception as exc:
                log.append(f"手部识别失败，身体动作仍可使用：{exc}")

        save_results_meta(output_dir, files)
        progress(1.0, desc="完成")
        log.append("处理完成。")
        return files["pt"], files["npz"], files["video"], "\n".join(log)
    finally:
        INFERENCE_LOCK.release()


def build_ui():
    with gr.Blocks(title="GVHMR → MMD") as demo:
        state = gr.State(value=None)
        gr.Markdown(
            """
# GVHMR → MMD 动作提取
上传单人视频，在远程 GPU 上恢复世界坐标动作，并下载可由 Blender 4.5 插件直接读取的 `.npz`。
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="输入视频", sources=["upload"], height=400)
                with gr.Accordion("推理选项", open=False):
                    static_cam = gr.Checkbox(label="固定机位", value=False)
                    use_dpvo = gr.Checkbox(label="使用 DPVO", value=False)
                    focal_mm = gr.Slider(
                        label="等效焦距（mm，0=自动）",
                        minimum=0,
                        maximum=200,
                        value=0,
                        step=1,
                    )
                    export_npz = gr.Checkbox(
                        label="导出 Blender/MMD NPZ",
                        value=True,
                    )
                    render_preview = gr.Checkbox(
                        label="生成预览视频",
                        info="关闭后跳过耗时渲染，只输出姿态文件",
                        value=True,
                    )
                    recognize_hands = gr.Checkbox(
                        label="识别双手与手指动作",
                        info="使用腕部裁剪和 MediaPipe 三维手部关键点",
                        value=True,
                    )
                run_button = gr.Button("开始识别", variant="primary", size="lg")
            with gr.Column(scale=1):
                render_output = gr.Video(label="动作预览", height=400)
                npz_output = gr.File(label="下载 gvhmr_motion.npz（Blender 插件）")
                pt_output = gr.File(label="下载原始 hmr4d_results.pt")
                log_output = gr.Textbox(label="状态", lines=5, interactive=False)

        with gr.Accordion("模型权重", open=False):
            model_status = gr.Textbox(label="状态", interactive=False)
            download_button = gr.Button("检查 / 下载模型", size="sm")

        def on_run(
            video, static, dpvo, focal, npz, preview_enabled, hands_enabled,
            progress=gr.Progress(),
        ):
            pt, npz_file, preview, log = run_gvhmr(
                video,
                static,
                dpvo,
                focal,
                npz,
                preview_enabled,
                hands_enabled,
                progress=progress,
            )
            saved_state = {"pt": pt, "npz": npz_file, "video": preview}
            return preview, npz_file, pt, log, saved_state

        run_button.click(
            fn=on_run,
            inputs=[
                video_input,
                static_cam,
                use_dpvo,
                focal_mm,
                export_npz,
                render_preview,
                recognize_hands,
            ],
            outputs=[render_output, npz_output, pt_output, log_output, state],
        )

        def on_download():
            lines = []
            for line in download_models_yield():
                lines.append(line)
                yield "\n".join(lines)

        download_button.click(fn=on_download, outputs=model_status)

        def on_page_load():
            previous = find_latest_result()
            if previous and any(previous.values()):
                return (
                    previous.get("video"),
                    previous.get("npz"),
                    previous.get("pt"),
                    "已恢复上一次结果。",
                    previous,
                )
            return None, None, None, "", None

        demo.load(
            fn=on_page_load,
            outputs=[render_output, npz_output, pt_output, log_output, state],
        )
        gr.Markdown(
            """
---
建议：画面中只保留一名完整可见的人；固定相机请勾选“固定机位”；NPZ 不包含可执行 pickle 对象。
"""
        )
    return demo


if __name__ == "__main__":
    print(f"Missing model weights: {len(check_models_ready())}")
    host = os.environ.get("GVHMR_WEBUI_HOST", "127.0.0.1")
    port = int(os.environ.get("GVHMR_WEBUI_PORT", "7860"))
    username = os.environ.get("GVHMR_WEBUI_USERNAME")
    password = os.environ.get("GVHMR_WEBUI_PASSWORD")
    auth = (username, password) if username and password else None
    if host not in {"127.0.0.1", "localhost", "::1"} and auth is None:
        raise RuntimeError(
            "Refusing an unauthenticated non-local WebUI. Set both "
            "GVHMR_WEBUI_USERNAME and GVHMR_WEBUI_PASSWORD, or bind to 127.0.0.1."
        )
    app = build_ui()
    app.queue(max_size=3)
    app.launch(
        server_name=host,
        server_port=port,
        auth=auth,
        share=False,
        theme=gr.themes.Soft(),
    )
