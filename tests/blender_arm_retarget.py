"""Run with Blender --background --factory-startup --python this_file.py."""

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender_addon"))

from gvhmr_mmd_bridge.core import SMPL_PARENTS, MotionData, smpl_global_rotations
from gvhmr_mmd_bridge.mapping import BONE_RULES
from gvhmr_mmd_bridge.retarget import retarget_motion


def make_rig(upper_angle, lower_angle, helpers):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    data = bpy.data.armatures.new("RegressionRig")
    rig = bpy.data.objects.new("RegressionRig", data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    by_index = {rule.source_index: rule for rule in BONE_RULES}
    for rule in BONE_RULES:
        bone = data.edit_bones.new(rule.candidates[0])
        bone.head = (0, 0, 1 + rule.source_index * 0.1)
        bone.tail = bone.head + Vector((0, 0, 0.1))
    for rule in BONE_RULES:
        parent = int(SMPL_PARENTS[rule.source_index])
        while parent >= 0 and parent not in by_index:
            parent = int(SMPL_PARENTS[parent])
        if parent >= 0:
            data.edit_bones[rule.candidates[0]].parent = data.edit_bones[
                by_index[parent].candidates[0]
            ]
    rules = {rule.semantic: rule for rule in BONE_RULES}
    for side, sign in (("left", 1), ("right", -1)):
        upper = Vector((sign * np.cos(upper_angle), 0, -np.sin(upper_angle)))
        lower = Vector((sign * np.cos(lower_angle), 0, -np.sin(lower_angle)))
        shoulder = Vector((sign * 0.3, 0, 2))
        elbow = shoulder + upper
        wrist = elbow + lower
        for joint, head, tail in (
            ("shoulder", shoulder, elbow),
            ("elbow", elbow, wrist),
            ("wrist", wrist, wrist + lower * 0.3),
        ):
            bone = data.edit_bones[rules[f"{side}_{joint}"].candidates[0]]
            bone.head, bone.tail = head, tail
            bone.roll = sign * 0.37
        if helpers:
            arm = data.edit_bones[rules[f"{side}_shoulder"].candidates[0]]
            forearm = data.edit_bones[rules[f"{side}_elbow"].candidates[0]]
            helper = data.edit_bones.new(f"twist.{side}")
            helper.head = shoulder + upper * 0.5
            helper.tail = elbow
            helper.parent = arm
            forearm.parent = helper
            # A deliberately misleading display tail must not set calibration.
            arm.tail = shoulder + Vector((0, 0.3, 0))
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig, rules


def check(upper_angle, lower_angle, helpers=False, enabled=True, flip=False):
    rig, rules = make_rig(upper_angle, lower_angle, helpers)
    body = np.zeros((4, 21, 3))
    # Neutral, isolated bend, raised/twisted upper arm, and moving torso/wrist.
    for shoulder, elbow, wrist in ((16, 18, 20), (17, 19, 21)):
        body[1:, elbow - 1] = (0, 1.2, 0)
        body[2:, shoulder - 1] = (0.4, -0.3, 0.8)
        body[3, wrist - 1] = (0.2, 0.3, -0.1)
    body[3, 8] = (0.2, -0.1, 0.3)
    motion = MotionData(body, np.array([[0., 0., 0.]] * 3 + [[0.3, 0.2, -0.4]]),
                        np.zeros((4, 3)), 30)
    source = smpl_global_rotations(motion, flip_forward=flip)
    retarget_motion(rig, motion, auto_scale=False, apply_hands=False,
                    compensate_arm_rest_pose=enabled, flip_forward=flip)
    for frame in range(4):
        bpy.context.scene.frame_set(frame + 1)
        bpy.context.view_layer.update()
        for side, sign in (("left", 1), ("right", -1)):
            for joint, child in (("shoulder", "elbow"), ("elbow", "wrist")):
                rule = rules[f"{side}_{joint}"]
                bone = rig.pose.bones[rule.candidates[0]]
                end = rig.pose.bones[rules[f"{side}_{child}"].candidates[0]]
                actual = np.array((end.head - bone.head).normalized())
                rest = np.array((end.bone.head_local - bone.bone.head_local).normalized())
                desired = np.array((sign, 0, 0)) if enabled else rest
                expected = source[frame, rule.source_index] @ desired
                np.testing.assert_allclose(actual, expected, atol=2e-5,
                                           err_msg=f"frame={frame} {side} {joint}")
            # Wrist's global rest delta follows the calibrated forearm frame.
            wrist_rule = rules[f"{side}_wrist"]
            wrist = rig.pose.bones[wrist_rule.candidates[0]]
            direction = np.array((wrist.tail - wrist.head).normalized())
            rest = np.array((wrist.bone.tail_local - wrist.bone.head_local).normalized())
            expected = source[frame, wrist_rule.source_index] @ (
                np.array((sign, 0, 0)) if enabled else rest
            )
            np.testing.assert_allclose(direction, expected, atol=2e-5)


if __name__ == "__main__":
    for args in (
        (0.7, 0.7),
        (0.7, 0.4),
        (0., 0.),
        (0.7, 0.4, True),
        (0.7, 0.4, False, False),
        (0.7, 0.4, True, True, True),
    ):
        check(*args)
    print("PASS: 6 rigs, 24 frames, both arms and wrists (144 direction checks)")
