# Server integration

English instructions follow the Chinese section.

## 中文

本目录中的文件应复制到独立的官方 GVHMR checkout；本仓库不包含或再分发 GVHMR。

```bash
export BRIDGE_ROOT=/path/to/gvhmr-to-mmd-bridge
export GVHMR_ROOT=/path/to/GVHMR

cp "$BRIDGE_ROOT/server/gvhmr_export.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/webui.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/progress.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/hand_pose_mediapipe.py" "$GVHMR_ROOT/"
```

仅更新进度显示时，复制 `webui.py` 和 `progress.py` 后重启 WebUI 即可，
无需修改 GVHMR 代码。已有的可选 `gvhmr_demo_skip_render.patch` 仅用于让旧版上游
支持关闭预览渲染；不应用该补丁时，在不支持 `--skip_render` 的版本中保持预览渲染开启。

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

### 升级已有 WebUI

先在运行 WebUI 的终端按 `Ctrl+C` 停止旧进程，再在服务器执行以下命令。
将路径替换成实际位置；`BRIDGE_ROOT` 是本项目的 `main` 分支 checkout，`GVHMR_ROOT` 是
官方 GVHMR 目录。只更新本仓库、执行 `make build` 或安装 Blender 插件 ZIP，不会同步之前复制的文件。

```bash
conda activate gvhmr
export BRIDGE_ROOT=/path/to/gvhmr-to-mmd-bridge
export GVHMR_ROOT=/path/to/GVHMR

git -C "$BRIDGE_ROOT" pull --ff-only && \
cp "$BRIDGE_ROOT/server/webui.py" \
   "$BRIDGE_ROOT/server/progress.py" \
   "$BRIDGE_ROOT/server/gvhmr_export.py" \
   "$BRIDGE_ROOT/server/hand_pose_mediapipe.py" "$GVHMR_ROOT/" && \
cd "$GVHMR_ROOT" && \
python -c "import progress; from progress import GVHMRProgressParser, StageUpdate, iter_log_records; print(progress.__file__)" && \
python webui.py
```

导入检查应打印 GVHMR 目录下的 `progress.py` 路径，成功后才会启动 WebUI。
任何命令失败时，先解决报错再继续。没有在服务器克隆本项目时，也可从同一提交的源码中
上传这四个文件到 `GVHMR_ROOT`，然后在该目录执行导入检查和启动命令。
此次进度更新不需要重新安装模型、重新推理或修改官方 GVHMR 源码。

### 启动报错：No module named 'progress'

`ModuleNotFoundError: No module named 'progress'` 发生在 WebUI 导入阶段，尚未开始推理。
新版 `webui.py` 依赖本项目的 `progress.py` 来解析 GVHMR 日志。通常是更新时只复制了
`webui.py`，遗漏了配套文件，或将它放到了错误目录。

按上面的升级步骤更新文件，确保 `progress.py` 与实际启动的 `webui.py` 在同一目录，
通过导入检查后重启。**不要执行 `pip install progress`**：PyPI 同名包不是本项目模块。
如果文件已存在但仍报错，请在该目录执行上述导入检查，确认输出路径指向此处的 `progress.py`，
并检查使用的 Python 环境是否正确。

## English

Copy these files into a separate official GVHMR checkout. This repository does not contain or
redistribute GVHMR.

```bash
export BRIDGE_ROOT=/path/to/gvhmr-to-mmd-bridge
export GVHMR_ROOT=/path/to/GVHMR

cp "$BRIDGE_ROOT/server/gvhmr_export.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/webui.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/progress.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/hand_pose_mediapipe.py" "$GVHMR_ROOT/"
```

For a progress-only update, replace `webui.py` and `progress.py` and restart the WebUI.
No GVHMR patch is required. The existing optional `gvhmr_demo_skip_render.patch` concerns only
disabling preview rendering on older upstream versions; keep previews enabled when your
unpatched GVHMR does not support `--skip_render`.

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

### Upgrade an existing WebUI

Stop the old WebUI with `Ctrl+C` in its terminal, then run these commands on the server.
Replace the paths: `BRIDGE_ROOT` is this project's checkout on `main`, and `GVHMR_ROOT` is
the official GVHMR directory. Pulling this repository, running `make build`, or installing
the Blender ZIP does not synchronize files previously copied into GVHMR.

```bash
conda activate gvhmr
export BRIDGE_ROOT=/path/to/gvhmr-to-mmd-bridge
export GVHMR_ROOT=/path/to/GVHMR

git -C "$BRIDGE_ROOT" pull --ff-only && \
cp "$BRIDGE_ROOT/server/webui.py" \
   "$BRIDGE_ROOT/server/progress.py" \
   "$BRIDGE_ROOT/server/gvhmr_export.py" \
   "$BRIDGE_ROOT/server/hand_pose_mediapipe.py" "$GVHMR_ROOT/" && \
cd "$GVHMR_ROOT" && \
python -c "import progress; from progress import GVHMRProgressParser, StageUpdate, iter_log_records; print(progress.__file__)" && \
python webui.py
```

The import check should print the path to `progress.py` inside GVHMR; the WebUI starts only
if the check succeeds. Resolve any command failure before continuing. If this project is
not cloned on the server, upload these four files from the same source commit into
`GVHMR_ROOT`, then run the import check and startup command there. This progress update
requires no model reinstallation, new inference run, or edits to official GVHMR source.

### Startup error: No module named 'progress'

`ModuleNotFoundError: No module named 'progress'` occurs while importing the WebUI, before
inference starts. The updated `webui.py` uses this project's `progress.py` to parse GVHMR
logs. Usually only `webui.py` was copied, or its companion was placed in a different directory.

Follow the upgrade steps above, placing `progress.py` beside the `webui.py` you actually
run. Pass the import check and restart. **Do not run `pip install progress`**: the package
with that name on PyPI is not this project's module. If the file exists but startup still
fails, run the import check in that directory, confirm it prints the local `progress.py` path,
and verify the Python environment you are using.

See the [Chinese manual](../docs/USER_GUIDE.zh-CN.md) or
[English manual](../docs/USER_GUIDE.en.md) for the complete workflow.
