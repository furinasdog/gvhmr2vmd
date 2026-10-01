# GVHMR to MMD Bridge

中文 · [English](README.md) · [中文完整手册](docs/USER_GUIDE.zh-CN.md) · [Full English Manual](docs/USER_GUIDE.en.md)

GVHMR to MMD Bridge 将 [GVHMR](https://github.com/zju3dv/GVHMR) 恢复的人体动作和
MediaPipe 手部关键点转换为安全的 NumPy `.npz`，再在 Blender 4.5 中重定向到标准
MMD 骨架。

主要功能：

- 世界坐标身体动作、根位移与 30 FPS 时间线；
- 标准日文 MMD 骨骼及 mmd_tools 的 `.L/.R` 重命名；
- MMD A-Pose 与 SMPL T-Pose 的上臂静止角自动补偿；
- 左右 30 根手指骨骼、逐帧置信度和短缺帧平滑；
- 足部 IK 临时静音及原状态恢复；
- 不含 pickle 对象的 NPZ v2，插件端不依赖 PyTorch；
- 本地绑定、可通过 SSH 隧道访问的 Gradio WebUI。

快速开始：

1. 按[中文完整手册](docs/USER_GUIDE.zh-CN.md)部署 GVHMR 服务器集成。
2. 执行 `python tools/package_addon.py` 生成 `dist/gvhmr_mmd_bridge.zip`。
3. 在 Blender 4.5 中从磁盘安装 ZIP，并用 mmd_tools 导入 PMX。
4. 在“GVHMR MMD”面板选择 NPZ 和 Armature，检查映射后应用动作。

> [!IMPORTANT]
> 本仓库代码采用 GPL-3.0-or-later，但 GVHMR 上游许可证仅允许教育、研究和非营利用途。
> SMPL、SMPL-X、MANO、模型权重以及 PMX 模型均有各自许可，且不包含在本仓库中。
> 详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
