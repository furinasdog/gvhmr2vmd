"""Blender operators exposed by the add-on."""

from __future__ import annotations

import traceback

import bpy

from .core import load_motion
from .mapping import BONE_RULES, resolve_bone_map
from .retarget import restore_ik_constraints, retarget_motion


class GVHMRMMD_OT_validate(bpy.types.Operator):
    bl_idname = "gvhmr_mmd.validate"
    bl_label = "检查文件与骨骼映射"
    bl_description = "验证 NPZ，并显示可识别的 MMD 主骨骼数量"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = context.scene.gvhmr_mmd
        try:
            motion = load_motion(bpy.path.abspath(props.motion_path))
            if props.armature is None or props.armature.type != "ARMATURE":
                raise ValueError("请选择一个 MMD Armature")
            mapping, missing = resolve_bone_map(b.name for b in props.armature.data.bones)
            if missing:
                raise ValueError("缺少必需骨骼: " + ", ".join(missing))
            props.status = (
                f"✓ {motion.frame_count} 帧 @ {motion.fps:g} FPS；"
                f"识别 {len(mapping)}/{len(BONE_RULES)} 根主骨骼"
            )
            self.report({"INFO"}, props.status)
            return {"FINISHED"}
        except Exception as exc:
            props.status = f"错误：{exc}"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class GVHMRMMD_OT_apply(bpy.types.Operator):
    bl_idname = "gvhmr_mmd.apply"
    bl_label = "应用 GVHMR 动作"
    bl_description = "创建新 Action 并将 GVHMR 动作重定向到所选 MMD 骨架"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.gvhmr_mmd
        try:
            motion = load_motion(bpy.path.abspath(props.motion_path))
            result = retarget_motion(
                props.armature,
                motion,
                start_frame=props.start_frame,
                auto_scale=props.auto_scale,
                manual_scale=props.manual_scale,
                flip_forward=props.flip_forward,
                sync_fps=props.sync_fps,
                disable_ik=props.disable_ik,
                compensate_arm_rest_pose=props.compensate_arm_rest_pose,
                apply_hands=props.apply_hands,
                hand_confidence_threshold=props.hand_confidence_threshold,
                limit_finger_splay=props.limit_finger_splay,
                limit_finger_angles=props.limit_finger_angles,
                finger_max_flexion=props.finger_max_flexion,
                finger_max_extension=props.finger_max_extension,
            )
            optional = len(result.missing_optional)
            props.status = (
                f"✓ 已创建 {result.action_name}：{result.frame_count} 帧，"
                f"{result.mapped_bones} 根骨骼，位移比例 {result.translation_scale:.4g}"
            )
            if optional:
                props.status += f"（跳过 {optional} 根可选骨骼）"
            if result.keyed_hand_frames:
                props.status += (
                    f"；手指 {result.mapped_hand_bones}/30 根，"
                    f"有效手帧 {result.keyed_hand_frames}"
                )
            self.report({"INFO"}, props.status)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            props.status = f"错误：{exc}"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class GVHMRMMD_OT_restore_ik(bpy.types.Operator):
    bl_idname = "gvhmr_mmd.restore_ik"
    bl_label = "恢复 IK 约束"
    bl_description = "恢复由本插件静音前的 IK 约束状态"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.gvhmr_mmd
        if props.armature is None or props.armature.type != "ARMATURE":
            self.report({"ERROR"}, "请选择一个 MMD Armature")
            return {"CANCELLED"}
        count = restore_ik_constraints(props.armature)
        props.status = f"已恢复 {count} 个 IK 约束状态"
        self.report({"INFO"}, props.status)
        return {"FINISHED"}
