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
2. 从 [Releases](https://github.com/furinasdog/gvhmr2vmd/releases) 下载 `gvhmr_mmd_bridge.zip`，或执行 `make build` 在 `dist/` 中构建。
3. 在 Blender 4.5 中从磁盘安装 ZIP，并用 mmd_tools 导入 PMX。
4. 在“GVHMR MMD”面板选择 NPZ 和 Armature，检查映射后应用动作。

> [!IMPORTANT]
> 本仓库代码采用 GPL-3.0-or-later，但 GVHMR 上游许可证仅允许教育、研究和非营利用途。
> SMPL、SMPL-X、MANO、模型权重以及 PMX 模型均有各自许可，且不包含在本仓库中。
> 详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 自动发布

**Build and release** 工作流执行 `make build`，校验 ZIP 内容和 Python 语法，
然后将插件 ZIP 与 `SHA256SUMS` 发布到 GitHub Release，无需额外配置令牌或服务器依赖。

- 推送 `v0.2.0` 这类版本标签，自动构建并发布该提交。
- 也可在 **Actions → Build and release → Run workflow** 选择分支或标签，
  填写发布标签后手动执行。新标签指向本次构建的提交；已有标签必须指向同一提交。
- `v0.3.0-beta.1` 这类带后缀的标签发布为预发布版本。已发布的 Release 不会被覆盖；
  上传失败且 Release 仍为草稿时，可以重新运行。
