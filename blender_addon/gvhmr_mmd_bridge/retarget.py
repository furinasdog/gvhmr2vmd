"""Blender-side retargeting from SMPL rotations to an MMD armature."""

from __future__ import annotations

import math
import json
from dataclasses import dataclass

import bpy
from mathutils import Matrix, Quaternion, Vector

from .core import MotionData, blender_translation, smpl_global_rotations
from .mapping import BONE_RULES, RULE_BY_SEMANTIC, resolve_bone_map
from .hand_retarget import retarget_hands


IK_STATE_KEY = "gvhmr_mmd_previous_mute"
NOMINAL_SMPL_HEIGHT_METERS = 1.70


@dataclass(frozen=True)
class RetargetResult:
    action_name: str
    frame_count: int
    mapped_bones: int
    missing_optional: tuple[str, ...]
    translation_scale: float
    mapped_hand_bones: int = 0
    keyed_hand_frames: int = 0


def validate_armature(armature) -> tuple[dict[str, str], list[str]]:
    if armature is None or armature.type != "ARMATURE":
        raise ValueError("请选择一个 MMD Armature")
    resolved, missing = resolve_bone_map(bone.name for bone in armature.data.bones)
    if missing:
        raise ValueError("缺少必需骨骼: " + ", ".join(missing))
    return resolved, missing


def _mapped_parent_semantic(armature, bone_name: str, reverse_map: dict[str, str]):
    bone = armature.data.bones[bone_name].parent
    while bone is not None:
        semantic = reverse_map.get(bone.name)
        if semantic is not None:
            return semantic
        bone = bone.parent
    return None


def _estimate_height(armature, mapping: dict[str, str]) -> float:
    points = []
    for semantic in (
        "head",
        "neck",
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
    ):
        name = mapping.get(semantic)
        if name:
            bone = armature.data.bones[name]
            points.extend((bone.head_local.z, bone.tail_local.z))
    if len(points) < 4:
        for name in mapping.values():
            bone = armature.data.bones[name]
            points.extend((bone.head_local.z, bone.tail_local.z))
    height = max(points) - min(points) if points else 0.0
    if height <= 1e-6:
        raise ValueError("无法从骨架估算模型高度，请使用手动根位移比例")
    return height


def _arm_rest_corrections(armature, mapping: dict[str, str]) -> dict[str, Quaternion]:
    """Align an MMD A-pose upper arm with the SMPL zero-pose T-pose.

    The correction is expressed in armature space and is applied after the
    source shoulder rotation. Children inherit it naturally through Blender's
    pose hierarchy, so elbow and wrist motion remain relative to the arm.
    """
    corrections = {}
    for semantic, desired in (
        ("left_shoulder", Vector((1.0, 0.0, 0.0))),
        ("right_shoulder", Vector((-1.0, 0.0, 0.0))),
    ):
        bone = armature.data.bones[mapping[semantic]]
        rest_direction = bone.tail_local - bone.head_local
        if rest_direction.length <= 1e-8:
            continue
        corrections[semantic] = rest_direction.normalized().rotation_difference(desired)
    return corrections


def mute_ik_constraints(armature) -> int:
    if IK_STATE_KEY not in armature:
        previous_state = []
        for pose_bone in armature.pose.bones:
            for constraint in pose_bone.constraints:
                if constraint.type == "IK":
                    previous_state.append(
                        {
                            "bone": pose_bone.name,
                            "constraint": constraint.name,
                            "mute": bool(constraint.mute),
                        }
                    )
        armature[IK_STATE_KEY] = json.dumps(previous_state, ensure_ascii=False)

    changed = 0
    for pose_bone in armature.pose.bones:
        for constraint in pose_bone.constraints:
            if constraint.type != "IK":
                continue
            if not constraint.mute:
                constraint.mute = True
                changed += 1
    return changed


def restore_ik_constraints(armature) -> int:
    if IK_STATE_KEY not in armature:
        return 0
    try:
        previous_state = json.loads(armature[IK_STATE_KEY])
    except (TypeError, ValueError, json.JSONDecodeError):
        previous_state = []

    restored = 0
    for item in previous_state:
        pose_bone = armature.pose.bones.get(item.get("bone", ""))
        if pose_bone is None:
            continue
        constraint = pose_bone.constraints.get(item.get("constraint", ""))
        if constraint is None or constraint.type != "IK":
            continue
        constraint.mute = bool(item.get("mute", False))
        restored += 1
    del armature[IK_STATE_KEY]
    return restored


def _action_name(motion: MotionData) -> str:
    base = motion.source_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if base:
        base = base.rsplit(".", 1)[0]
    return f"GVHMR_{base or 'Motion'}"


def retarget_motion(
    armature,
    motion: MotionData,
    *,
    start_frame: int = 1,
    auto_scale: bool = True,
    manual_scale: float = 1.0,
    flip_forward: bool = False,
    sync_fps: bool = True,
    disable_ik: bool = True,
    compensate_arm_rest_pose: bool = True,
    apply_hands: bool = True,
    hand_confidence_threshold: float = 0.35,
) -> RetargetResult:
    mapping, _ = validate_armature(armature)
    optional_semantics = {rule.semantic for rule in BONE_RULES if not rule.required}
    missing_optional = tuple(sorted(optional_semantics.difference(mapping)))

    if disable_ik:
        mute_ik_constraints(armature)

    if auto_scale:
        translation_scale = _estimate_height(armature, mapping) / NOMINAL_SMPL_HEIGHT_METERS
    else:
        if manual_scale <= 0:
            raise ValueError("手动根位移比例必须大于零")
        translation_scale = float(manual_scale)

    scene = bpy.context.scene
    if sync_fps:
        rounded_fps = max(1, int(round(motion.fps)))
        scene.render.fps = rounded_fps
        scene.render.fps_base = rounded_fps / motion.fps
        frame_step = 1.0
    else:
        scene_fps = scene.render.fps / scene.render.fps_base
        frame_step = scene_fps / motion.fps

    armature.animation_data_create()
    action = bpy.data.actions.new(_action_name(motion))
    armature.animation_data.action = action

    source_global = smpl_global_rotations(motion, flip_forward=flip_forward)
    root_translation = blender_translation(motion, flip_forward=flip_forward)
    reverse_map = {bone_name: semantic for semantic, bone_name in mapping.items()}
    parent_semantics = {
        semantic: _mapped_parent_semantic(armature, bone_name, reverse_map)
        for semantic, bone_name in mapping.items()
    }
    rest_rotations = {
        semantic: armature.data.bones[bone_name].matrix_local.to_quaternion()
        for semantic, bone_name in mapping.items()
    }
    rest_pose_corrections = (
        _arm_rest_corrections(armature, mapping)
        if compensate_arm_rest_pose
        else {}
    )

    previous_quaternions: dict[str, Quaternion] = {}
    ordered = [rule for rule in BONE_RULES if rule.semantic in mapping]

    for frame_index in range(motion.frame_count):
        target_frame = start_frame + frame_index * frame_step
        for rule in ordered:
            semantic = rule.semantic
            bone_name = mapping[semantic]
            pose_bone = armature.pose.bones[bone_name]
            parent_semantic = parent_semantics[semantic]
            joint_rotation = source_global[frame_index, rule.source_index]
            if parent_semantic is not None:
                parent_index = RULE_BY_SEMANTIC[parent_semantic].source_index
                parent_rotation = source_global[frame_index, parent_index]
                joint_rotation = parent_rotation.T @ joint_rotation

            rest = rest_rotations[semantic]
            motion_quaternion = Matrix(joint_rotation.tolist()).to_quaternion()
            correction = rest_pose_corrections.get(semantic)
            if correction is not None:
                motion_quaternion = motion_quaternion @ correction
            basis_quaternion = rest.inverted() @ motion_quaternion @ rest
            basis_quaternion.normalize()
            previous = previous_quaternions.get(bone_name)
            if previous is not None:
                basis_quaternion.make_compatible(previous)
            previous_quaternions[bone_name] = basis_quaternion.copy()

            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = basis_quaternion
            pose_bone.keyframe_insert(
                data_path="rotation_quaternion",
                frame=target_frame,
                group=bone_name,
            )

            if semantic == "root":
                armature_space = Vector(
                    (root_translation[frame_index] * translation_scale).tolist()
                )
                pose_bone.location = rest.inverted() @ armature_space
                pose_bone.keyframe_insert(
                    data_path="location",
                    frame=target_frame,
                    group=bone_name,
                )

    mapped_hand_bones = keyed_hand_frames = 0
    if apply_hands and (
        motion.left_hand_landmarks is not None or motion.right_hand_landmarks is not None
    ):
        mapped_hand_bones, keyed_hand_frames = retarget_hands(
            armature,
            motion,
            mapping,
            start_frame=start_frame,
            frame_step=frame_step,
            confidence_threshold=hand_confidence_threshold,
        )

    for fcurve in action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"

    scene.frame_start = min(scene.frame_start, start_frame)
    last_frame = int(math.ceil(start_frame + (motion.frame_count - 1) * frame_step))
    scene.frame_end = max(scene.frame_end, last_frame)
    scene.frame_set(start_frame)

    return RetargetResult(
        action_name=action.name,
        frame_count=motion.frame_count,
        mapped_bones=len(mapping),
        missing_optional=missing_optional,
        translation_scale=translation_scale,
        mapped_hand_bones=mapped_hand_bones,
        keyed_hand_frames=keyed_hand_frames,
    )
