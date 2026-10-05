"""Blender --background --factory-startup --python tests/blender_finger_limits.py."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender_addon"))
from gvhmr_mmd_bridge.hand_retarget import _flexion_only, _palm_basis, retarget_hands
from gvhmr_mmd_bridge.mapping import HAND_BONE_RULES

# Pure side motion must disappear, while flexion and extension survive.
rest = Vector((0, 1, 0))
normal = Vector((0, 0, 1))
for angle in (-0.8, 0., 0.8):
    side = _flexion_only(Quaternion(normal, angle), rest, normal)
    assert abs(side.angle) < 1e-6
    bend = Quaternion((1, 0, 0), angle)
    actual = _flexion_only(bend, rest, normal)
    assert ((actual @ rest) - (bend @ rest)).length < 1e-6

# Mirrored hinges must clamp flexion and extension on the correct side.
for sign in (1., -1.):
    axis = normal.cross(rest).normalized()
    for degrees, expected in ((150, 90), (-100, -10), (45, 45), (-5, -5), (0, 0)):
        delta = Quaternion(axis, sign * math.radians(degrees))
        limited = _flexion_only(delta, rest, normal, flexion_sign=sign,
                                angle_limits=(-math.radians(10), math.radians(90)))
        desired = Quaternion(axis, sign * math.radians(expected)) @ rest
        assert ((limited @ rest) - desired).length < 1e-6
    locked = _flexion_only(Quaternion(axis, 1), rest, normal,
                           flexion_sign=sign, angle_limits=(0, 0))
    assert locked.angle < 1e-6

data = bpy.data.armatures.new("FingerLimitTest")
rig = bpy.data.objects.new("FingerLimitTest", data)
bpy.context.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
body_mapping = {}
hand_mapping = {r.semantic: r.candidates[0] for r in HAND_BONE_RULES}
for side, sign in (("left", 1), ("right", -1)):
    wrist = data.edit_bones.new(f"wrist.{side}")
    wrist.head = (sign * 3, 0, 0)
    wrist.tail = (sign * 3, 0.5, 0)
    body_mapping[f"{side}_wrist"] = wrist.name
    for rule in HAND_BONE_RULES:
        if rule.side != side:
            continue
        finger = rule.semantic.split("_")[1][:-1]
        index = int(rule.semantic[-1]) - 1
        offset = {"thumb": 1.5, "index": 1, "middle": 0, "ring": -0.5, "little": -1}[finger]
        direction = Vector((sign * 0.5 if finger == "thumb" else 0, 1, 0)).normalized()
        bone = data.edit_bones.new(hand_mapping[rule.semantic])
        bone.head = wrist.head + Vector((sign * offset, 1, 0)) + index * direction
        bone.tail = bone.head + direction
        bone.roll = sign * (0.2 + index * 0.4)
        bone.parent = data.edit_bones[hand_mapping[rule.parent_semantic]] if index else wrist
bpy.ops.object.mode_set(mode="OBJECT")
motion = SimpleNamespace(frame_count=3)
axes = {}
for side in ("left", "right"):
    palm = _palm_basis(rig, body_mapping, hand_mapping, side)
    landmarks = np.zeros((3, 21, 3))
    for rule in HAND_BONE_RULES:
        if rule.side != side:
            continue
        bone = data.bones[hand_mapping[rule.semantic]]
        rest = (bone.tail_local - bone.head_local).normalized()
        hinge = palm.col[2].cross(rest).normalized()
        axes[bone.name] = bone.matrix_local.to_quaternion().inverted() @ hinge
        start, end = rule.landmark_pair
        for frame, amount in enumerate((0.3, 0.7, -0.4)):
            delta = Quaternion(palm.col[2], 0.35) @ Quaternion(
                hinge, amount * int(rule.semantic[-1])
            )
            direction = palm.inverted() @ (delta @ rest)
            landmarks[frame, end] = landmarks[frame, start] + np.array(direction)
    setattr(motion, f"{side}_hand_landmarks", landmarks)
    setattr(motion, f"{side}_hand_confidence", np.array([1., 0., 1.]))

rig.animation_data_create()
for limited, angle_limited in ((True, True), (True, False), (False, True)):
    action = bpy.data.actions.new(f"Limited={limited}")
    rig.animation_data.action = action
    result = retarget_hands(rig, motion, body_mapping, start_frame=1, frame_step=1,
                            confidence_threshold=0.35, limit_finger_splay=limited,
                            limit_finger_angles=angle_limited,
                            finger_max_flexion=0.2, finger_max_extension=0.1)
    assert result == (30, 4), result
    assert len(action.fcurves) == 120
    assert all(len(curve.keyframe_points) == 2 for curve in action.fcurves)
    off_axis = 0
    moving = 0
    for frame in (1, 3):
        bpy.context.scene.frame_set(frame)
        for name, axis in axes.items():
            q = rig.pose.bones[name].rotation_quaternion
            vector = Vector((q.x, q.y, q.z))
            moving += vector.length > 1e-4
            off_axis += vector.cross(axis).length > 1e-5
            if limited and angle_limited:
                side = next(r.side for r in HAND_BONE_RULES if hand_mapping[r.semantic] == name)
                angle = 2 * math.atan2(vector.dot(axis), q.w)
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                flexion = angle * (1 if side == "left" else -1)
                assert -0.1 - 1e-6 <= flexion <= 0.2 + 1e-6, (name, flexion)
    assert moving > 0
    assert off_axis == 0 if limited else off_axis > 0
print("PASS: both hands, 30 fingers, flexion/extension, bone roll, confidence and opt-out")
