"""Scene properties for the add-on UI."""

import math

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


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
    source_start: IntProperty(
        name="源起始帧", description="从 NPZ 的第几帧开始（从 1 计数，包含此帧）",
        default=1, min=1,
    )
    source_end: IntProperty(
        name="源结束帧", description="包含此帧；0 表示直到文件末尾",
        default=0, min=0,
    )
    speed: FloatProperty(
        name="播放速度", description="2 为两倍速，0.5 为半速；身体和手指同步变速",
        default=1.0, min=0.1, max=4.0,
    )
    rotation_smoothing: FloatProperty(
        name="身体旋转平滑（秒）",
        description="居中时间窗口降低身体和整体朝向的抖动；0 关闭，过大会削弱快速动作",
        default=0.0, min=0.0, max=0.5, precision=3,
    )
    translation_smoothing: FloatProperty(
        name="根位移平滑（秒）",
        description="平滑整体移动轨迹；0 关闭，不会锁定脚部接触点",
        default=0.0, min=0.0, max=0.5, precision=3,
    )
    root_motion: EnumProperty(
        name="根位移模式",
        items=(
            ("FULL", "完整位移", "保留前后、左右和上下移动"),
            ("IN_PLACE", "原地（保留高度）", "移除水平移动，保留跳跃和上下起伏"),
            ("NONE", "固定位置", "移除全部根位移，保留身体旋转"),
        ),
        default="FULL",
    )
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
        description="将整体朝向和位移轨迹绕骨架 Z 轴旋转 180°，保持关节相对动作；需重新应用动作",
        default=False,
    )
    compensate_arm_rest_pose: BoolProperty(
        name="校正 MMD 手臂静止角度",
        description="按骨骼位置校正 A-Pose 上臂与前臂，并同步转换肘部和手腕的旋转坐标系",
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
    limit_finger_splay: BoolProperty(
        name="手指仅前后屈伸",
        description="限制所有指节（含拇指）的侧向摆动和扭转，保留模型初始张开角度",
        default=True,
    )
    limit_finger_angles: BoolProperty(
        name="限制屈伸角度",
        description="在仅前后屈伸模式下，限制每个指节相对模型初始姿态的屈伸幅度（含拇指）",
        default=True,
    )
    finger_max_flexion: FloatProperty(
        name="最大内弯",
        description="每个指节向掌心方向弯曲的最大角度；应按模型调整",
        subtype="ANGLE", default=math.radians(90), min=0.0, max=math.pi,
    )
    finger_max_extension: FloatProperty(
        name="最大后伸",
        description="每个指节向手背方向伸展的最大角度；设为 0 可禁止后伸",
        subtype="ANGLE", default=math.radians(10), min=0.0, max=math.pi,
    )
    hand_confidence_threshold: FloatProperty(
        name="手部置信度阈值",
        description="低于此置信度的手部帧不会写关键帧",
        default=0.35,
        min=0.0,
        max=1.0,
    )
    status: StringProperty(name="状态", default="")
