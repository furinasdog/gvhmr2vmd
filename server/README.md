# Server integration

English instructions follow the Chinese section.

## 中文

本目录中的文件应复制到独立的官方 GVHMR checkout；本仓库不包含或再分发 GVHMR。

```bash
export BRIDGE_ROOT=/path/to/gvhmr-to-mmd-bridge
export GVHMR_ROOT=/path/to/GVHMR

cp "$BRIDGE_ROOT/server/gvhmr_export.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/webui.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/hand_pose_mediapipe.py" "$GVHMR_ROOT/"
git -C "$GVHMR_ROOT" apply "$BRIDGE_ROOT/server/patches/gvhmr_demo_skip_render.patch"
```

在 GVHMR Python 3.10 环境安装：

```bash
python -m pip install gradio==6.26.0
python -m pip install -r "$BRIDGE_ROOT/server/requirements-hand.txt"
python -m pip install --no-deps mediapipe==0.10.14
```

SMPL 和 SMPL-X 必须从官方网站接受许可后手动放入 GVHMR 指定目录。本 WebUI 不会从
第三方镜像下载这些文件。启动：

```bash
cd "$GVHMR_ROOT"
python webui.py
```

WebUI 默认仅监听 `127.0.0.1:7860`。推荐使用 SSH 隧道：

```bash
ssh -L 7860:127.0.0.1:7860 -p <ssh-port> <user>@<server-host>
```

非本机监听必须同时设置 `GVHMR_WEBUI_USERNAME` 和 `GVHMR_WEBUI_PASSWORD`。

## English

Copy these files into a separate official GVHMR checkout. This repository does not contain or
redistribute GVHMR.

```bash
export BRIDGE_ROOT=/path/to/gvhmr-to-mmd-bridge
export GVHMR_ROOT=/path/to/GVHMR

cp "$BRIDGE_ROOT/server/gvhmr_export.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/webui.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/hand_pose_mediapipe.py" "$GVHMR_ROOT/"
git -C "$GVHMR_ROOT" apply "$BRIDGE_ROOT/server/patches/gvhmr_demo_skip_render.patch"
```

Install inside the GVHMR Python 3.10 environment:

```bash
python -m pip install gradio==6.26.0
python -m pip install -r "$BRIDGE_ROOT/server/requirements-hand.txt"
python -m pip install --no-deps mediapipe==0.10.14
```

Obtain SMPL and SMPL-X from their official sites after accepting their terms, then place them at
the paths required by GVHMR. The WebUI intentionally does not download them from mirrors.

```bash
cd "$GVHMR_ROOT"
python webui.py
```

The default bind is `127.0.0.1:7860`. Prefer an SSH tunnel:

```bash
ssh -L 7860:127.0.0.1:7860 -p <ssh-port> <user>@<server-host>
```

A non-local bind requires both `GVHMR_WEBUI_USERNAME` and `GVHMR_WEBUI_PASSWORD`.

See the [Chinese manual](../docs/USER_GUIDE.zh-CN.md) or
[English manual](../docs/USER_GUIDE.en.md) for the complete workflow.
