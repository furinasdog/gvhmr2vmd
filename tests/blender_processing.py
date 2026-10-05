"""Blender --background --factory-startup --python tests/blender_processing.py."""

import json
import math
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender_addon"))
sys.path.insert(0, str(ROOT / "tests"))

from blender_arm_retarget import make_rig  # noqa: E402
from gvhmr_mmd_bridge import register, unregister  # noqa: E402
from gvhmr_mmd_bridge.core import MotionData  # noqa: E402
from gvhmr_mmd_bridge.retarget import IK_STATE_KEY, retarget_motion  # noqa: E402


def check_baked_controls():
    rig, rules = make_rig(0.7, 0.4, helpers=True)
    motion = MotionData(
        np.zeros((7, 21, 3)), np.zeros((7, 3)),
        np.column_stack((np.arange(7), np.arange(7) * 0.2, np.arange(7) * 0.4)),
        30.0,
    )
    root_name = rules["root"].candidates[0]
    old_action = bpy.data.actions.new("Keep this action")
    rig.animation_data_create()
    rig.animation_data.action = old_action
    settings = dict(auto_scale=False, apply_hands=False, start_frame=10,
                    source_start=2, source_end=6, speed=2.0)
    for sync_fps in (True, False):
        for root_motion in ("FULL", "IN_PLACE", "NONE"):
            scene = bpy.context.scene
            scene.render.fps = 24
            scene.render.fps_base = 1.0
            result = retarget_motion(rig, motion, sync_fps=sync_fps,
                                     root_motion=root_motion, **settings)
            assert result.frame_count == 5
            scene_fps = 30.0 if sync_fps else 24.0
            assert math.isclose(scene.render.fps / scene.render.fps_base, scene_fps)
            end = 10 + 4 * scene_fps / 60
            action = rig.animation_data.action
            np.testing.assert_allclose(action.frame_range, [10, end], atol=1e-5)
            assert json.loads(action["gvhmr_mmd_settings"])["source_end"] == 6
            scene.frame_set(math.floor(end), subframe=end - math.floor(end))
            bpy.context.view_layer.update()
            bone = rig.pose.bones[root_name]
            actual = bone.bone.matrix_local.to_quaternion() @ bone.location
            expected = {"FULL": [4, -1.6, 0.8], "IN_PLACE": [0, 0, 0.8],
                        "NONE": [0, 0, 0]}[root_motion]
            np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert old_action.use_fake_user
    retarget_motion(
        rig, motion, auto_scale=False, apply_hands=False, source_start=np.int64(2),
        source_end=np.int64(6), speed=np.float32(2), rotation_smoothing=np.float32(0.1),
        translation_smoothing=np.float32(0.1),
    )
    metadata = json.loads(rig.animation_data.action["gvhmr_mmd_settings"])
    assert metadata["source_start"] == 2 and metadata["source_end"] == 6
    assert metadata["speed"] == 2.0
    # Rejected inputs must leave the current action, timeline and IK untouched.
    constraint = rig.pose.bones[root_name].constraints.new("IK")
    constraint.mute = False
    if IK_STATE_KEY in rig:
        del rig[IK_STATE_KEY]
    action = rig.animation_data.action
    actions_count = len(bpy.data.actions)
    frame = bpy.context.scene.frame_current
    for extra in ({"source_start": 8}, {"speed": 0}, {"manual_scale": float("nan")}):
        invalid = settings | extra
        try:
            retarget_motion(rig, motion, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid settings were accepted")
        assert rig.animation_data.action == action
        assert len(bpy.data.actions) == actions_count
        assert bpy.context.scene.frame_current == frame
        assert not constraint.mute
        assert IK_STATE_KEY not in rig
    print("PASS: crop, speed, root modes, prior action preservation and invalid settings")


def check_operator_single_frame():
    register()
    try:
        rig, _ = make_rig(0.7, 0.4, helpers=False)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "single.npz"
            np.savez(source, format_version=2, body_pose=np.zeros((1, 21, 3)),
                     global_orient=np.zeros((1, 3)), transl=np.zeros((1, 3)), fps=29.97)
            props = bpy.context.scene.gvhmr_mmd
            props.motion_path = str(source)
            props.armature = rig
            props.auto_scale = False
            props.speed = 0.5
            props.rotation_smoothing = 0.1
            props.translation_smoothing = 0.1
            props.root_motion = "IN_PLACE"
            assert bpy.ops.gvhmr_mmd.validate() == {"FINISHED"}
            assert bpy.ops.gvhmr_mmd.apply() == {"FINISHED"}
            assert rig.animation_data.action is not None
            scene = bpy.context.scene
            assert math.isclose(scene.render.fps / scene.render.fps_base, 29.97,
                                rel_tol=1e-6)
    finally:
        unregister()
    print("PASS: registered UI properties and single-frame validate/apply operators")


check_baked_controls()
check_operator_single_frame()
