# GVHMR MMD Motion NPZ v2

文件使用 `numpy.savez_compressed` 写入，读取时禁止 pickle。v2 增加可选的双手三维关键点；
Blender 插件仍兼容不含手部数据的 v1。

身体必需字段：

| 字段 | dtype / shape | 说明 |
| --- | --- | --- |
| `format_version` | `int32 scalar` | 当前为 `2`；读取方兼容 `1` |
| `body_pose` | `float32 [F,21,3]` | SMPL-X 身体局部 axis-angle，弧度 |
| `global_orient` | `float32 [F,3]` | 骨盆全局旋转 |
| `transl` | `float32 [F,3]` | 骨盆全局位移，单位米 |
| `fps` | `float32 scalar` | 动作帧率，范围 `[1,240]` |

手部可选字段：

| 字段 | dtype / shape | 说明 |
| --- | --- | --- |
| `left_hand_landmarks` | `float32 [F,21,3]` | 左手 21 点，规范化手掌坐标 |
| `right_hand_landmarks` | `float32 [F,21,3]` | 右手 21 点，规范化手掌坐标 |
| `left_hand_confidence` | `float32 [F]` | 左手逐帧置信度，范围 `[0,1]` |
| `right_hand_confidence` | `float32 [F]` | 右手逐帧置信度，范围 `[0,1]` |
| `hand_backend` | Unicode scalar | 当前为 `mediapipe-hand-landmarker` |
| `hand_coordinate_system` | Unicode scalar | 当前为 `canonical_palm_v1` |

规范化手掌坐标以腕点为原点，X 指向拇指侧，Y 从腕指向中指 MCP，Z 由右手系确定，
并以腕到中指 MCP 的距离归一化。关键点编号遵循 MediaPipe 21 点顺序。

其余可选字段包括 `betas`、`static_confidence`、`K_fullimg`、`source_name` 和
`coordinate_system`。读取方必须检查有限数、形状、帧数及置信度范围。

每侧手部关键点与置信度必须成对出现，置信度必须为一维 `[F]`。身体帧数至少为 1；
单帧 `[1,21,3]` 是合法动作。兼容的身体输入也包括 `[F,63]`、`[1,F,63]`、
`[1,F,21,3]`，根旋转和位移兼容 `[1,F,3]`；其他轴顺序不会被自动重排。
