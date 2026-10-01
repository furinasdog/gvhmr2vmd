# Third-party notices / 第三方说明

This file is informational and is not legal advice. Always read the upstream terms before use.

本文仅作信息说明，不构成法律意见。使用前请阅读各上游项目的完整许可条款。

## Project source / 本项目源码

The original source in this repository is licensed under GPL-3.0-or-later. The license does not
relicense third-party software, model files, checkpoints, videos, or MMD assets.

本仓库原创源码采用 GPL-3.0-or-later。该许可不会改变第三方软件、模型文件、检查点、
视频或 MMD 素材原有的许可。

## GVHMR

- Upstream: <https://github.com/zju3dv/GVHMR>
- Upstream terms permit use, copying, modification, and distribution for educational, research,
  and non-profit purposes only. Commercial use requires separate permission from its authors.
- GVHMR is not vendored by this repository. The small patch in `server/patches/` must be applied
  by the user to a separately obtained checkout and remains subject to the upstream terms.

- 上游：<https://github.com/zju3dv/GVHMR>
- 上游条款仅允许教育、研究和非营利用途；商业用途需要另行取得作者许可。
- 本仓库不包含 GVHMR。`server/patches/` 中的小补丁由用户应用到独立 checkout，相关
  使用仍受 GVHMR 上游条款约束。

## MediaPipe and Hand Landmarker

- Upstream: <https://github.com/google-ai-edge/mediapipe>
- MediaPipe source is Apache-2.0. The Hand Landmarker task is downloaded from Google's official
  model storage on first use. Review the model card and terms published with the selected model.
- No MediaPipe source or task model is committed to this repository.

- 上游：<https://github.com/google-ai-edge/mediapipe>
- MediaPipe 源码采用 Apache-2.0。Hand Landmarker 首次使用时从 Google 官方模型存储下载；
  使用者仍应阅读对应模型卡和条款。
- 本仓库不提交 MediaPipe 源码或 task 模型文件。

## Blender and mmd_tools

- Blender: <https://www.blender.org/> — GPL.
- mmd_tools: <https://github.com/MMD-Blender/blender_mmd_tools> — GPL-3.0.
- They are external runtime dependencies and are not redistributed here.

- Blender：<https://www.blender.org/> — GPL。
- mmd_tools：<https://github.com/MMD-Blender/blender_mmd_tools> — GPL-3.0。
- 二者均为外部运行时依赖，本仓库不再分发。

## SMPL, SMPL-X, MANO, HaMeR, and checkpoints

- SMPL, SMPL-X, and MANO require registration and acceptance of their own licenses. Users must
  obtain them through official channels. They must never be committed to this repository.
- HaMeR is not a runtime dependency in version 0.2.0. An optional future backend would still
  require a separately licensed MANO model and review of HaMeR's then-current terms.
- GVHMR, HMR2, ViTPose, YOLO, DPVO, and other checkpoints remain under their upstream terms.

- SMPL、SMPL-X 和 MANO 需要注册并接受各自许可，必须通过官方渠道取得，禁止提交到
  本仓库。
- 0.2.0 不依赖 HaMeR。未来若添加可选后端，仍需要单独授权的 MANO 文件并重新核对
  HaMeR 当时的许可。
- GVHMR、HMR2、ViTPose、YOLO、DPVO 等检查点遵循各自上游条款。

## PMX models, textures, video, and generated motion

No character model, texture, test video, or generated motion is part of the project distribution.
Users are responsible for the terms of every input and output asset. Do not assume that public
download availability grants redistribution or commercial rights.

任何角色模型、贴图、测试视频和生成动作都不属于本项目发行内容。使用者应自行确认每个
输入和输出素材的条款；可公开下载不等于允许再分发或商用。
