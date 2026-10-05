"""Regression for shoulder P/C controls; run with Blender --background.

This synthetic rig reproduces a cancel-helper mechanism, not any particular
PMX asset: P -> deform shoulder -> inverse-P C -> upper arm. Driving the actual
shoulder preserves its motion, while selecting P causes C to cancel it.
"""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender_addon"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blender_arm_retarget import make_rig
from gvhmr_mmd_bridge import retarget
from gvhmr_mmd_bridge.core import MotionData, smpl_global_rotations
from gvhmr_mmd_bridge.mapping import resolve_bone_map


def make_control_rig():
    rig, rules = make_rig(0.7, 0.4, True)
    bpy.ops.object.mode_set(mode="EDIT")
    for side, prefix in (("left", "左"), ("right", "右")):
        shoulder = rig.data.edit_bones[rules[f"{side}_collar"].candidates[0]]
        upper = rig.data.edit_bones[rules[f"{side}_shoulder"].candidates[0]]
        control = rig.data.edit_bones.new(f"{prefix}肩P")
        control.head, control.tail = shoulder.head.copy(), shoulder.tail.copy()
        control.roll = shoulder.roll
        control.parent = shoulder.parent
        shoulder.parent = control
        cancel = rig.data.edit_bones.new(f"{prefix}肩C")
        cancel.head, cancel.tail = control.head.copy(), control.tail.copy()
        cancel.roll = control.roll
        cancel.parent = shoulder
        upper.parent = cancel
        wrist = rig.data.edit_bones[rules[f"{side}_wrist"].candidates[0]]
        wrist_helper = rig.data.edit_bones.new(f"wrist_twist.{side}")
        wrist_helper.head = wrist.head - Vector((0.03, 0.01, 0.02))
        wrist_helper.tail = wrist.head.copy()
        wrist_helper.roll = -0.48
        wrist_helper.parent = wrist.parent
        wrist.parent = wrist_helper
    bpy.ops.object.mode_set(mode="OBJECT")
    for prefix in ("左", "右"):
        constraint = rig.pose.bones[f"{prefix}肩C"].constraints.new("COPY_ROTATION")
        constraint.name = "Inverse shoulder P"
        constraint.target = rig
        constraint.subtarget = f"{prefix}肩P"
        constraint.owner_space = "LOCAL"
        constraint.target_space = "LOCAL"
        constraint.invert_x = constraint.invert_y = constraint.invert_z = True
    return rig, rules


def make_motion():
    body = np.zeros((5, 21, 3))
    for side, sign, collar, shoulder, elbow, wrist in (
        ("left", 1, 13, 16, 18, 20), ("right", -1, 14, 17, 19, 21)
    ):
        # Single-axis P motion makes the helper's inverse exact. It varies per
        # frame, so this verifies evaluated constraints rather than a rest rig.
        body[:, collar - 1, 1] = sign * np.array([0.0, 0.5, 0.7, -0.3, 0.2])
        body[2:, shoulder - 1] = (0.3, sign * -0.4, sign * 0.5)
        body[2:, elbow - 1] = (0.2, sign * 1.1, -0.2)
        body[3:, wrist - 1] = (0.3, 0.2, sign * -0.5)
    body[4, 8] = (0.2, -0.1, 0.3)
    orient = np.zeros((5, 3))
    orient[4] = (0.2, 0.3, -0.1)
    return MotionData(body, orient, np.zeros((5, 3)), 30)


def legacy_mapping(names):
    """Reproduce the previous P-first rule when both candidates are present."""
    result, missing = resolve_bone_map(names)
    for side, prefix in (("left", "左"), ("right", "右")):
        result[f"{side}_collar"] = f"{prefix}肩P"
    return result, missing


def maximum_direction_error(rig, rules, motion, flip):
    source = smpl_global_rotations(motion, flip_forward=flip)
    maximum = 0.0
    side_errors = {"left": 0.0, "right": 0.0}
    for frame in range(motion.frame_count):
        bpy.context.scene.frame_set(frame + 1)
        bpy.context.view_layer.update()
        for side, sign in (("left", 1), ("right", -1)):
            for joint, child in (("shoulder", "elbow"), ("elbow", "wrist")):
                rule = rules[f"{side}_{joint}"]
                bone = rig.pose.bones[rule.candidates[0]]
                end = rig.pose.bones[rules[f"{side}_{child}"].candidates[0]]
                actual = np.array((end.head - bone.head).normalized())
                expected = source[frame, rule.source_index] @ np.array((sign, 0, 0))
                error = float(np.linalg.norm(actual - expected))
                maximum = max(maximum, error)
                side_errors[side] = max(side_errors[side], error)
    return maximum, side_errors


def check_deform_mapping(flip):
    rig, rules = make_control_rig()
    motion = make_motion()
    mapping, missing = resolve_bone_map(b.name for b in rig.data.bones)
    assert not missing
    for side, prefix in (("left", "左"), ("right", "右")):
        assert mapping[f"{side}_collar"] == f"{prefix}肩"
    corrections = retarget._arm_rest_corrections(rig, mapping)
    retarget.retarget_motion(rig, motion, auto_scale=False, apply_hands=False, flip_forward=flip)
    maximum, _ = maximum_direction_error(rig, rules, motion, flip)
    assert maximum < 2e-5, maximum
    source = smpl_global_rotations(motion, flip_forward=flip)
    for frame in range(motion.frame_count):
        bpy.context.scene.frame_set(frame + 1)
        bpy.context.view_layer.update()
        for side in ("left", "right"):
            for joint in ("shoulder", "elbow", "wrist"):
                semantic = f"{side}_{joint}"
                rule = rules[semantic]
                bone = rig.pose.bones[rule.candidates[0]]
                actual = np.array(bone.matrix.to_3x3() @ bone.bone.matrix_local.to_3x3().inverted())
                expected = (
                    source[frame, rule.source_index] @ np.array(corrections[semantic].to_matrix())
                )
                np.testing.assert_allclose(actual, expected, atol=2e-5,
                                           err_msg=f"flip={flip} frame={frame} {semantic}")
    # Pose data confirms the P controls are untouched by the corrected mapping.
    for prefix in ("左", "右"):
        assert rig.pose.bones[f"{prefix}肩P"].matrix_basis.to_quaternion().angle < 1e-6


def check_legacy_failure():
    rig, rules = make_control_rig()
    motion = make_motion()
    with patch.object(retarget, "resolve_bone_map", legacy_mapping):
        retarget.retarget_motion(rig, motion, auto_scale=False, apply_hands=False)
    _, side_errors = maximum_direction_error(rig, rules, motion, False)
    assert all(error > 0.3 for error in side_errors.values()), side_errors
    # At frame 2 the source contains only a 0.5-radian collar turn. The old
    # selection loses it, leaving exactly that angular error in both arms.
    bpy.context.scene.frame_set(2)
    bpy.context.view_layer.update()
    source = smpl_global_rotations(motion)
    for side, sign in (("left", 1), ("right", -1)):
        shoulder = rig.pose.bones[rules[f"{side}_shoulder"].candidates[0]]
        elbow = rig.pose.bones[rules[f"{side}_elbow"].candidates[0]]
        actual = np.array((elbow.head - shoulder.head).normalized())
        expected = source[1, rules[f"{side}_shoulder"].source_index] @ np.array((sign, 0, 0))
        angle = math.acos(float(np.clip(actual @ expected, -1, 1)))
        assert abs(angle - 0.5) < 2e-5, (side, angle)


def check_reapply_resets_previous_controls():
    rig, rules = make_control_rig()
    motion = make_motion()
    with patch.object(retarget, "resolve_bone_map", legacy_mapping):
        retarget.retarget_motion(rig, motion, auto_scale=False, apply_hands=False)
    old_action = rig.animation_data.action
    original_curves = {
        (curve.data_path, curve.array_index): [tuple(key.co) for key in curve.keyframe_points]
        for curve in old_action.fcurves
    }
    bpy.context.scene.frame_set(3)
    bpy.context.view_layer.update()
    assert rig.pose.bones["左肩P"].matrix_basis.to_quaternion().angle > 0.5

    # New actions do not reset channels they omit. Reapply while the previous
    # action has left its P control posed, then switch actions again to ensure
    # an actual neutral key prevents that old control state from leaking back.
    retarget.retarget_motion(rig, motion, auto_scale=False, apply_hands=False)
    new_action = rig.animation_data.action
    maximum, _ = maximum_direction_error(rig, rules, motion, False)
    assert maximum < 2e-5, ("stale P control after reapply", maximum)
    assert old_action.use_fake_user
    assert original_curves == {
        (curve.data_path, curve.array_index): [tuple(key.co) for key in curve.keyframe_points]
        for curve in old_action.fcurves
    }
    for prefix in ("左", "右"):
        control = rig.pose.bones[f"{prefix}肩P"]
        keyed = [curve for curve in new_action.fcurves
                 if curve.data_path == control.path_from_id("rotation_quaternion")]
        assert len(keyed) == 4, (control.name, "neutral rotation channels must be keyed")
    rig.animation_data.action = old_action
    bpy.context.scene.frame_set(3)
    bpy.context.view_layer.update()
    assert rig.pose.bones["左肩P"].matrix_basis.to_quaternion().angle > 0.5
    rig.animation_data.action = new_action
    maximum, _ = maximum_direction_error(rig, rules, motion, False)
    assert maximum < 2e-5, ("stale P control after switching actions", maximum)
    for prefix in ("左", "右"):
        assert rig.pose.bones[f"{prefix}肩P"].matrix_basis.to_quaternion().angle < 1e-6


def check_casefold_helpers():
    rig, rules = make_control_rig()
    motion = make_motion()
    for old_name, new_name in (("左肩P", "shoulderp_l"), ("右肩P", "肩p.R")):
        rig.data.bones[old_name].name = new_name
        bone = rig.pose.bones[new_name]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0.3, 0.4, -0.2)
    result = retarget.retarget_motion(rig, motion, auto_scale=False, apply_hands=False)
    assert result.neutralized_shoulder_helpers == 2
    maximum, _ = maximum_direction_error(rig, rules, motion, False)
    assert maximum < 2e-5, ("case-insensitive shoulder helpers", maximum)


def check_mapping_fallbacks():
    for names, expected in (
        (["左肩", "左肩P", "右肩", "右肩P"], ("左肩", "右肩")),
        (["肩.L", "肩P.L", "肩.R", "肩P.R"], ("肩.L", "肩.R")),
        (["LeftShoulder", "shoulderP_L", "RightShoulder", "shoulderP_R"],
         ("LeftShoulder", "RightShoulder")),
        (["左肩P", "右肩P"], ("左肩P", "右肩P")),
    ):
        mapping, _ = resolve_bone_map(names)
        assert (mapping["left_collar"], mapping["right_collar"]) == expected


if __name__ == "__main__":
    check_mapping_fallbacks()
    check_legacy_failure()
    check_deform_mapping(False)
    check_deform_mapping(True)
    check_reapply_resets_previous_controls()
    check_casefold_helpers()
    print("PASS: P-first loses 28.65 degrees; deform shoulders preserve both arms and wrists")
