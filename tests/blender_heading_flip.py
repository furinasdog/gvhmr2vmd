"""Blender --background --factory-startup --python tests/blender_heading_flip.py.

Expected heading is defined geometrically, independently of the conversion
function: flip the evaluated baseline pose by a half-turn, preserving its shape.
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender_addon"))
sys.path.insert(0, str(ROOT / "tests"))

from blender_arm_retarget import make_rig
from gvhmr_mmd_bridge.core import (
    MotionData,
    SMPL_PARENTS,
    blender_translation,
    load_motion,
    smpl_global_rotations,
)
from gvhmr_mmd_bridge.retarget import retarget_motion


HALF_TURN = np.diag([-1., -1., 1.])


def check_source_rotations():
    motion = MotionData(
        np.zeros((4, 21, 3)),
        np.array([[0., 0., 0.], [0., math.pi, 0.],
                  [0., math.pi / 4, 0.], [0.3, -0.6, 0.2]]),
        np.array([[2., 3., 4.], [3., 4., 5.], [1., 2., 3.], [4., 1., 2.]]),
        30,
    )
    motion.body_pose[3] = np.random.default_rng(7).normal(0, 0.3, (21, 3))
    normal = smpl_global_rotations(motion)
    flipped = smpl_global_rotations(motion, flip_forward=True)
    # Identity must become a half-turn, and a half-turn must become identity.
    np.testing.assert_allclose(flipped[0, 0], HALF_TURN, atol=1e-12)
    np.testing.assert_allclose(flipped[1, 0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(flipped, HALF_TURN @ normal, atol=1e-12)
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            np.testing.assert_allclose(
                flipped[:, parent].transpose(0, 2, 1) @ flipped[:, joint],
                normal[:, parent].transpose(0, 2, 1) @ normal[:, joint],
                atol=1e-12,
            )
    # Translation rotates with the character, with height unchanged.
    expected = np.array([[0., 0., 0.], [-1., 1., 1.],
                         [1., -1., -1.], [-2., -2., -2.]])
    np.testing.assert_allclose(blender_translation(motion, flip_forward=True), expected)
    return motion


def snapshot(rig, frames):
    poses = []
    for frame in frames:
        whole = math.floor(frame)
        bpy.context.scene.frame_set(whole, subframe=frame - whole)
        bpy.context.view_layer.update()
        evaluated = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
        poses.append({
            bone.name: (np.array(bone.matrix), np.array(bone.matrix_basis))
            for bone in evaluated.pose.bones
        })
    return poses


def check_baked_motion(motion, frames, compensate=True):
    rig, rules = make_rig(0.7, 0.4, helpers=True)
    root_name = rules["root"].candidates[0]
    # Match the downward-pointing center bone from the reported MMD rig.
    bpy.ops.object.mode_set(mode="EDIT")
    center = rig.data.edit_bones[root_name]
    center.tail = center.head + Vector((0., 0., -0.1))
    bpy.ops.object.mode_set(mode="OBJECT")
    pivot = np.array(rig.data.bones[root_name].head_local)
    settings = dict(auto_scale=False, manual_scale=1.0, apply_hands=False,
                    compensate_arm_rest_pose=compensate)
    retarget_motion(rig, motion, flip_forward=False, **settings)
    normal = snapshot(rig, frames)
    retarget_motion(rig, motion, flip_forward=True, **settings)
    flipped = snapshot(rig, frames)
    for frame, original, turned in zip(frames, normal, flipped):
        for name, (matrix, basis) in original.items():
            target, target_basis = turned[name]
            label = f"frame={frame} bone={name}"
            np.testing.assert_allclose(target[:3, :3], HALF_TURN @ matrix[:3, :3],
                                       atol=3e-5, err_msg=label)
            np.testing.assert_allclose(target[:3, 3],
                                       pivot + HALF_TURN @ (matrix[:3, 3] - pivot),
                                       atol=3e-5, err_msg=label)
            if name != root_name:
                np.testing.assert_allclose(target_basis, basis, atol=3e-5, err_msg=label)
    print(f"PASS: baked heading flip, {motion.frame_count} source frames, "
          f"{len(frames)} sampled times, arm correction={compensate}")


synthetic = check_source_rotations()
for compensate in (False, True):
    check_baked_motion(synthetic, [1, 1.5, 2, 2.5, 3, 3.5, 4], compensate)

fixture = ROOT / "test_output" / "gvhmr_motion_dance_hands.npz"
if fixture.is_file():
    motion = load_motion(fixture)
    # Check every source frame and one interpolated time on the reported file.
    check_baked_motion(motion, [*range(1, motion.frame_count + 1), 97.5])
else:
    print("SKIP: optional local dance NPZ is unavailable")
print("PASS: heading correction changes the whole pose, not its relative joint motion")
