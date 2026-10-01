"""Retarget canonical 21-point hand landmarks to MMD finger bones."""

from __future__ import annotations

from mathutils import Matrix, Quaternion, Vector

from .mapping import HAND_BONE_RULES, resolve_hand_bone_map


def _palm_basis(armature, body_mapping, hand_mapping, side: str) -> Matrix:
    wrist = armature.data.bones[body_mapping[f"{side}_wrist"]].head_local
    index = armature.data.bones[hand_mapping[f"{side}_index1"]].head_local
    middle = armature.data.bones[hand_mapping[f"{side}_middle1"]].head_local
    little = armature.data.bones[hand_mapping[f"{side}_little1"]].head_local
    x_axis = index - little
    y_axis = middle - wrist
    if x_axis.length <= 1e-8 or y_axis.length <= 1e-8:
        raise ValueError(f"无法从 {side} 手指骨架建立手掌坐标系")
    x_axis.normalize()
    y_axis.normalize()
    z_axis = x_axis.cross(y_axis)
    if z_axis.length <= 1e-8:
        raise ValueError(f"{side} 手掌骨骼共线，无法计算旋转")
    z_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()
    return Matrix((x_axis, y_axis, z_axis)).transposed()


def retarget_hands(
    armature,
    motion,
    body_mapping: dict[str, str],
    *,
    start_frame: int,
    frame_step: float,
    confidence_threshold: float,
) -> tuple[int, int]:
    """Insert finger keys into the currently assigned action.

    Returns ``(mapped_bones, keyed_hand_frames)``. A side is skipped when its
    landmarks or the three palm-reference finger roots are unavailable.
    """
    hand_mapping, _missing = resolve_hand_bone_map(b.name for b in armature.data.bones)
    mapped_count = len(hand_mapping)
    keyed_frames = 0
    previous_quaternions: dict[str, Quaternion] = {}

    for side in ("left", "right"):
        landmarks = getattr(motion, f"{side}_hand_landmarks")
        confidence = getattr(motion, f"{side}_hand_confidence")
        required = {f"{side}_index1", f"{side}_middle1", f"{side}_little1"}
        if landmarks is None or confidence is None or not required.issubset(hand_mapping):
            continue
        palm_basis = _palm_basis(armature, body_mapping, hand_mapping, side)
        side_rules = [
            rule for rule in HAND_BONE_RULES
            if rule.side == side and rule.semantic in hand_mapping
        ]
        rest_rotations = {
            rule.semantic: armature.data.bones[hand_mapping[rule.semantic]].matrix_local.to_quaternion()
            for rule in side_rules
        }
        rest_directions = {
            rule.semantic: (
                armature.data.bones[hand_mapping[rule.semantic]].tail_local
                - armature.data.bones[hand_mapping[rule.semantic]].head_local
            ).normalized()
            for rule in side_rules
        }

        for frame_index in range(motion.frame_count):
            if confidence[frame_index] < confidence_threshold:
                continue
            global_deltas = {}
            valid_frame = True
            for rule in side_rules:
                start, end = rule.landmark_pair
                direction = Vector((landmarks[frame_index, end] - landmarks[frame_index, start]).tolist())
                if direction.length <= 1e-8:
                    valid_frame = False
                    break
                desired = palm_basis @ direction.normalized()
                global_deltas[rule.semantic] = rest_directions[rule.semantic].rotation_difference(desired)
            if not valid_frame:
                continue

            target_frame = start_frame + frame_index * frame_step
            for rule in side_rules:
                bone_name = hand_mapping[rule.semantic]
                joint_delta = global_deltas[rule.semantic]
                if rule.parent_semantic in global_deltas:
                    joint_delta = global_deltas[rule.parent_semantic].inverted() @ joint_delta
                rest = rest_rotations[rule.semantic]
                basis = rest.inverted() @ joint_delta @ rest
                basis.normalize()
                previous = previous_quaternions.get(bone_name)
                if previous is not None:
                    basis.make_compatible(previous)
                previous_quaternions[bone_name] = basis.copy()
                pose_bone = armature.pose.bones[bone_name]
                pose_bone.rotation_mode = "QUATERNION"
                pose_bone.rotation_quaternion = basis
                pose_bone.keyframe_insert(
                    data_path="rotation_quaternion",
                    frame=target_frame,
                    group=bone_name,
                )
            keyed_frames += 1
    return mapped_count, keyed_frames
