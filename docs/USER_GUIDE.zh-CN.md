# GVHMR to MMD Bridge 完整操作手册

本文覆盖从服务器部署、视频推理、NPZ 下载，到 Blender 4.5 应用动作和导出 VMD 的
完整流程。命令中的路径均为示例，请替换为自己的绝对路径。

## 1. 工作流程

```text
单人视频
  ├─ GVHMR：身体姿态、世界坐标根位移、相机运动
  ├─ ViTPose 腕/肘缓存 → MediaPipe：左右手 21 点三维姿态
  └─ gvhmr_motion.npz（安全数值数组）
                         ↓
Blender 4.5 + mmd_tools → MMD Armature → Action → 可选导出 VMD
```

插件不会修改网格、材质或刚体。头发、衣服和饰品仍由 MMD/Blender 物理或手工处理。

## 2. 系统要求

### 推理服务器

- Linux；
- NVIDIA CUDA 显卡；
- Python 3.10；
- 已能正常运行的官方 GVHMR；
- GVHMR 所需的合法 SMPL、SMPL-X 和公开检查点；
- 建议至少 12 GB 显存和 20 GB 可用磁盘空间。

### 本地动画端

- Blender 4.5 LTS；
- 与 Blender 4.5 匹配的 mmd_tools 4.x；
- 本插件生成的 ZIP；
- 合法取得、允许用于当前作品的 PMX 模型。

ARP 不是必需依赖。标准 MMD 骨骼优先使用本插件的直接 FK 重定向；只有特殊骨架需要
额外中间绑定时才考虑 ARP。

## 3. 许可与文件准备

先阅读 [第三方说明](../THIRD_PARTY_NOTICES.md)。尤其注意：

- GVHMR 上游只授权教育、研究和非营利用途；
- SMPL/SMPL-X 必须从官方渠道注册、接受许可并自行下载；
- 不要提交 PMX、视频、检查点、SMPL 文件、NPZ 推理结果或 MANO 文件；
- 本仓库不会自动从第三方镜像下载 SMPL/SMPL-X。

建议目录：

```text
/work/GVHMR/                 官方 GVHMR checkout
/work/gvhmr-to-mmd-bridge/   本仓库 checkout
```

## 4. 部署服务器集成

### 4.1 安装并验证 GVHMR

先严格按照 GVHMR 官方文档创建环境。示例：

```bash
git clone https://github.com/zju3dv/GVHMR.git /work/GVHMR
cd /work/GVHMR
conda create -n gvhmr python=3.10
conda activate gvhmr
python -m pip install -r requirements.txt
python -m pip install -e .
```

将自行取得的文件放入官方要求的位置：

```text
inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```

其余公开检查点也应按 GVHMR 官方目录结构放置。先用官方示例视频确认 GVHMR 本身可用，
再接入本项目；否则很难区分上游环境问题和桥接问题。

### 4.2 复制集成文件

```bash
export BRIDGE_ROOT=/work/gvhmr-to-mmd-bridge
export GVHMR_ROOT=/work/GVHMR

cp "$BRIDGE_ROOT/server/gvhmr_export.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/webui.py" "$GVHMR_ROOT/"
cp "$BRIDGE_ROOT/server/hand_pose_mediapipe.py" "$GVHMR_ROOT/"
git -C "$GVHMR_ROOT" apply "$BRIDGE_ROOT/server/patches/gvhmr_demo_skip_render.patch"
```

### 4.3 安装 WebUI 与手部依赖

在 GVHMR 环境中执行：

```bash
conda activate gvhmr
python -m pip install gradio
python -m pip install -r "$BRIDGE_ROOT/server/requirements-hand.txt"
python -m pip install --no-deps mediapipe==0.10.14
```

固定版本是为了避免新版 MediaPipe 自动升级 NumPy/OpenCV，破坏 GVHMR 的 Ultralytics
依赖。安装后验证：

```bash
python - <<'PY'
import cv2, mediapipe, numpy, torch, ultralytics
print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
print("mediapipe", mediapipe.__version__)
print("ultralytics", ultralytics.__version__)
print("cuda", torch.cuda.is_available())
PY
```

经过验证的组合为 NumPy 1.23.5、OpenCV 4.11.0.86、MediaPipe 0.10.14。若你的官方
GVHMR 环境使用其他版本，优先保持 GVHMR 可用，并为手部模块建立独立环境。

## 5. 启动和访问 WebUI

```bash
cd /work/GVHMR
conda activate gvhmr
python webui.py
```

默认地址为 `127.0.0.1:7860`，不会直接暴露到公网。在本地电脑建立 SSH 隧道：

```bash
ssh -L 7860:127.0.0.1:7860 -p <SSH端口> <用户>@<服务器地址>
```

然后浏览器打开 `http://127.0.0.1:7860`。

如必须监听非本机地址：

```bash
export GVHMR_WEBUI_HOST=0.0.0.0
export GVHMR_WEBUI_PORT=7860
export GVHMR_WEBUI_USERNAME='your-user'
export GVHMR_WEBUI_PASSWORD='use-a-long-random-password'
python webui.py
```

非本机监听但未设置用户名和密码时，WebUI 会拒绝启动。不要把密码写进仓库。

## 6. 视频推理

### 推荐输入

- 单人、全身尽量完整可见；
- 手部至少有数十像素宽，避免严重运动模糊；
- 固定机位时勾选“固定机位”；
- 尽量保留原始帧率和画面，不要重复压缩；
- 多人画面当前不保证选择正确人物。

### WebUI 选项

- **固定机位**：给静止相机视频使用；
- **使用 DPVO**：可选，速度更慢且需要额外依赖；
- **等效焦距**：已知相机焦距时填写，否则保持 0；
- **导出 Blender/MMD NPZ**：通常保持开启；
- **生成预览视频**：关闭可跳过耗时渲染；
- **识别双手与手指动作**：使用腕/肘裁剪和 MediaPipe 21 点手部模型。

第一次手部推理会下载 Google 官方 `hand_landmarker.task`。推理结束后下载：

- `gvhmr_motion.npz`：Blender 插件使用；
- `hmr4d_results.pt`：GVHMR 原始结果，仅用于调试；
- 预览视频：仅在启用渲染时生成。

NPZ 只含数值和字符串数组，插件使用 `allow_pickle=False` 读取。格式见
[MOTION_FORMAT.md](MOTION_FORMAT.md)。

## 7. 纯命令行导出

已有官方 `hmr4d_results.pt` 时：

```bash
cd /work/GVHMR
python gvhmr_export.py outputs/demo/example/hmr4d_results.pt \
  -o outputs/demo/example/gvhmr_motion.npz --fps 30
```

为现有 NPZ 添加手部结果：

```bash
python hand_pose_mediapipe.py \
  --video outputs/demo/example/0_input_video.mp4 \
  --vitpose outputs/demo/example/preprocess/vitpose.pt \
  --model inputs/checkpoints/mediapipe/hand_landmarker.task \
  --motion-npz outputs/demo/example/gvhmr_motion.npz
```

手部脚本优先使用 GVHMR 标准化后的 `0_input_video.mp4`，从而与 ViTPose 帧序列对齐。

## 8. 构建和安装 Blender 插件

在本仓库根目录执行：

```bash
python tools/package_addon.py
```

输出为 `dist/gvhmr_mmd_bridge.zip`。安装步骤：

1. 打开 Blender 4.5；
2. 进入“编辑 → 偏好设置 → 插件”；
3. 选择“从磁盘安装”；
4. 选择 ZIP 并启用 **Animation: GVHMR to MMD Bridge**；
5. 更新插件时先关闭 Blender，覆盖安装后重新启动，避免旧模块缓存。

## 9. 导入模型和应用动作

1. 使用 mmd_tools 导入 PMX；
2. 不要先修改 Armature 的静止姿态；
3. 在 3D 视图按 `N` 打开侧栏，进入 **GVHMR MMD**；
4. 选择 `gvhmr_motion.npz`；
5. 选择模型的 Armature，而不是网格或 MMD 根 Empty；
6. 点击“检查文件与骨骼映射”；
7. 确认帧数和骨骼数后点击“应用 GVHMR 动作”。

### 选项说明

- **起始帧**：动作开始帧；
- **使用动作帧率**：将场景同步到 NPZ FPS；
- **自动根位移比例**：按骨架高度将米换算为模型单位；
- **反转朝向 180°**：人物整体背向时使用；
- **校正 MMD 手臂静止角度**：自动补偿 MMD A-Pose 与 SMPL T-Pose；
- **手指仅前后屈伸**：默认开启，限制所有指节（含拇指）的侧摆和扭转，保留模型初始张开角度；关闭可恢复完整手指旋转。更改后需重新应用动作。
- **限制屈伸角度**：在“手指仅前后屈伸”下默认开启，每个指节（含拇指）相对初始姿态最大内弯 90°、后伸 10°。可按模型调整；最大后伸设为 0°可禁止后伸。限制写入新动作关键帧，修改设置后需重新应用动作；这不是手指碰撞检测。
- **应用手指动作**：NPZ 有手部数据时驱动 30 根手指骨；
- **手部置信度阈值**：默认 0.35；抖动多时提高，缺帧多时降低；
- **静音 IK 约束**：防止 MMD 足 IK 覆盖腿部 FK 动作。

插件每次应用都会新建并分配一个 Action。旧 Action 不会自动删除，便于比较和回退。

## 10. 后期处理与 VMD

- 在 Graph Editor 中检查脚滑、手部抖动和遮挡段；
- 长时间看不到手时，删除错误手指关键帧并手工保持上一姿态；
- 需要恢复模型原 IK 状态时点击“恢复 IK 约束”；
- 物理骨骼应在动作确定后再烘焙；
- 使用 mmd_tools 导出 VMD 前，确认当前 Action、帧范围和模型骨骼命名正确。

GVHMR 不提供面部表情，插件也不生成表情、口型、头发或裙摆动作。

## 11. 常见问题

### 提示缺少 `腕.L`、`足首.R` 等骨骼

安装最新版插件。mmd_tools 会把原始 `左腕` 等名称改为 Blender 的 `.L/.R` 名称，
0.2.0 已同时支持两套命名。仍失败时确认选择的是 Armature。

### 手臂整体偏低或偏高

标准 MMD A-Pose 保持“校正 MMD 手臂静止角度”开启。目标本身是 T-Pose 时关闭。

### 手指不动

检查 NPZ 是否为 v2、WebUI 日志是否输出左右手检测率，并确认“应用手指动作”开启。
远景或严重模糊会降低检测率。

### 手指抖动或偶尔翻转

提高置信度阈值；在 Graph Editor 删除短暂错误关键帧。手掌遮挡或左右手交叉仍可能造成
误识别，这是当前单人轻量后端的限制。

### 人物滑动或比例错误

检查模型导入比例。先使用自动比例；若异常，关闭自动比例并逐步调整手动比例。

### WebUI 无法启动

运行 `python tools/webui_smoke_test.py`，检查权重、Gradio 和端口。公网监听必须配置认证。

### MediaPipe 安装破坏 NumPy

恢复 GVHMR 要求的 NumPy/OpenCV，然后按 4.3 节用 `--no-deps` 安装固定版 MediaPipe。
