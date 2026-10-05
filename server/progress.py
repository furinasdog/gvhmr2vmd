"""Read existing GVHMR logs; never estimate a task-wide completion percentage.

Log markers follow the official zju3dv/GVHMR tools/demo/demo.py and preproc
modules. This observer does not inject code into, or patch, GVHMR.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass

ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
COUNTER = re.compile(r"(?:^|\s)(\d[\d,]*)/(\d[\d,]*)\s*(?:\[|$)")
TQDM_STAGES = {
    "Copy": ("copy", "视频准备"),
    "YoloV8 Tracking": ("tracking", "预处理 · 人体跟踪"),
    "ViTPose": ("pose", "预处理 · 姿态关键点"),
    "HMR2 Feature": ("features", "预处理 · 图像特征"),
    "DPVO": ("camera", "预处理 · 相机运动"),
    "Rendering Incam": ("render_incam", "预览渲染 · 相机视角"),
    "Rendering Global": ("render_global", "预览渲染 · 全局视角"),
}


def clean_log_line(line: str) -> str:
    return ANSI_ESCAPE.sub("", line).strip()


def iter_log_records(stream, raw_log=None):
    """Yield CR/LF records promptly, retaining every original byte in raw_log."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    read = getattr(stream, "read1", stream.read)
    while True:
        chunk = read(4096)
        if not chunk:
            pending += decoder.decode(b"", final=True)
            break
        if raw_log is not None:
            raw_log.write(chunk)
            raw_log.flush()
        pending += decoder.decode(chunk)
        records = re.split(r"[\r\n]", pending)
        pending = records.pop()
        for record in records:
            cleaned = clean_log_line(record)
            if cleaned:
                yield cleaned
    if pending:
        cleaned = clean_log_line(pending)
        if cleaned:
            yield cleaned


@dataclass(frozen=True)
class StageUpdate:
    stage: str
    description: str
    completed: int = 0
    total: int | None = None

    def report(self, progress) -> None:
        description = f"当前阶段：{self.description}"
        value = (0, None)
        if self.total is not None:
            description += f" · {self.completed}/{self.total}"
            if self.completed < self.total:
                value = (self.completed, self.total)
                description += "（仅本阶段进度）"
            else:
                # A finished tqdm loop does not prove file writing/process success.
                description += "（循环结束，等待阶段完成）"
        else:
            description += "（未提供总进度）"
        progress(value, desc=description)


class GVHMRProgressParser:
    """Conservatively recognize explicit stage evidence from official logs."""

    def __init__(self):
        self.current_stage = ""
        self.inference_started = False
        self.history = []
        self._last_description = None

    def _update(self, stage, description, line="", *, count=False):
        completed, total = 0, None
        if count and (match := COUNTER.search(line)):
            completed, total = (int(part.replace(",", "")) for part in match.groups())
            if total <= 0 or completed > total:
                completed, total = 0, None
        self.current_stage = stage
        if description != self._last_description:
            self.history.append(description)
            self._last_description = description
        return StageUpdate(stage, description, completed, total)

    def parse(self, line: str) -> StageUpdate | None:
        line = clean_log_line(line)
        for marker, (stage, description) in TQDM_STAGES.items():
            if line.startswith(marker + ":"):
                return self._update(stage, description, line, count=True)
        if "[Preprocess] Start" in line:
            return self._update("preprocess", "预处理 · 初始化")
        for marker, stage, description in (
            ("bbx (xyxy, xys) from", "tracking", "预处理 · 人体跟踪"),
            ("vitpose from", "pose", "预处理 · 姿态关键点"),
            ("vit_features from", "features", "预处理 · 图像特征"),
            ("slam results from", "camera", "预处理 · 相机运动"),
        ):
            if "[Preprocess] " + marker in line:
                return self._update(stage, description + "（读取缓存）")
        if "[Preprocess] End" in line:
            return self._update("preprocess_done", "预处理结束 · 等待 GVHMR")
        if "[HMR4D] Predicting" in line:
            self.inference_started = True
            return self._update("inference", "GVHMR 推理")
        if "[HMR4D] Elapsed" in line:
            return self._update("inference_save", "GVHMR 推理结束 · 保存姿态结果")
        if "[SimpleVO]" in line:
            return self._update("camera", "预处理 · 相机运动")
        if self.current_stage == "camera" and re.match(r"^\d+(?:\.\d+)?%\|", line):
            return self._update("camera", "预处理 · 相机运动", line, count=True)
        for marker, stage, description in (
            ("[Render Incam] Video already exists", "render_incam", "预览渲染 · 相机视角"),
            ("[Render Global] Video already exists", "render_global", "预览渲染 · 全局视角"),
        ):
            if marker in line:
                return self._update(stage, description + "（读取缓存）")
        if "[Render] Skipped" in line:
            return self._update("render_skipped", "预览渲染已跳过")
        if "[Merge Videos]" in line:
            return self._update("merge", "合并预览视频")
        if "[Copy Video]" in line:
            return self._update("copy", "视频准备（检查或转换）")
        return None
