"""Scene properties for the add-on UI."""

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty


def _armature_poll(_self, obj):
    return obj is not None and obj.type == "ARMATURE"


class GVHMRMMDProperties(bpy.types.PropertyGroup):
    motion_path: StringProperty(
        name="GVHMR 动作",
        description="WebUI 导出的 gvhmr_motion.npz",
        subtype="FILE_PATH",
    )
    armature: PointerProperty(
        name="MMD 骨架",
        description="由 mmd_tools 导入的 Armature",
        type=bpy.types.Object,
        poll=_armature_poll,
    )
    start_frame: IntProperty(name="起始帧", default=1, min=-100000, max=100000)
    sync_fps: BoolProperty(
        name="使用动作帧率",
        description="把场景帧率设为 NPZ 中的帧率；关闭时会按当前场景帧率重采样时间",
        default=True,
    )
    auto_scale: BoolProperty(
        name="自动根位移比例",
        description="根据 MMD 主骨架高度把 GVHMR 的米单位换算为模型单位",
        default=True,
    )
    manual_scale: FloatProperty(
        name="手动比例",
        description="关闭自动比例时使用的根位移倍率",
        default=1.0,
        min=0.0001,
        soft_max=10.0,
    )
    flip_forward: BoolProperty(
        name="反转朝向 180°",
        description="当角色整体背对预期方向时启用",
        default=False,
    )
    compensate_arm_rest_pose: BoolProperty(
        name="校正 MMD 手臂静止角度",
        description="把 MMD 的斜向下 A-Pose 上臂自动对齐到 GVHMR/SMPL 的 T-Pose",
        default=True,
    )
    disable_ik: BoolProperty(
        name="静音 IK 约束",
        description="避免 MMD 足 IK 覆盖 GVHMR 写入的 FK 腿部动作",
        default=True,
    )
    apply_hands: BoolProperty(
        name="应用手指动作",
        description="NPZ 包含手部识别结果时，为 MMD 手指骨骼生成关键帧",
        default=True,
    )
    hand_confidence_threshold: FloatProperty(
        name="手部置信度阈值",
        description="低于此置信度的手部帧不会写关键帧",
        default=0.35,
        min=0.0,
        max=1.0,
    )
    status: StringProperty(name="状态", default="")
