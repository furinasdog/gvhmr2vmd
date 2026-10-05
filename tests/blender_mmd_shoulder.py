"""Optional real-model regression; pass --pmx and --motion after Blender's --.

Run with --background --factory-startup --python-exit-code 1. Requires an installed
mmd_tools extension. No model or motion assets are bundled or modified.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender_addon"))

from gvhmr_mmd_bridge.core import load_motion, smpl_global_rotations  # noqa: E402
from gvhmr_mmd_bridge.mapping import resolve_bone_map  # noqa: E402
from gvhmr_mmd_bridge.retarget import (  # noqa: E402
    _arm_rest_corrections,
    retarget_motion,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pmx", required=True)
parser.add_argument("--motion", required=True)
parser.add_argument("--mmd-tools-module", default="bl_ext.blender_org.mmd_tools")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
bpy.ops.preferences.addon_enable(module=args.mmd_tools_module)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mmd_tools.import_model(
    filepath=args.pmx, types={"ARMATURE"}, scale=0.08, rename_bones=True,
    save_log=False, log_level="ERROR",
)
rig = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
motion = load_motion(args.motion)
mapping, missing = resolve_bone_map(b.name for b in rig.data.bones)
assert not missing, missing
rest = {b.name: np.array(b.matrix_local) for b in rig.data.bones}
corrections = _arm_rest_corrections(rig, mapping)
# Reproduce nonzero controller rotations left over by an earlier import.
for suffix in ("L", "R"):
    helper = rig.pose.bones.get(f"肩P.{suffix}")
    if helper:
        helper.rotation_mode = "XYZ"
        helper.rotation_euler = (0.23, -0.35, 0.51)
result = retarget_motion(rig, motion, flip_forward=True, auto_scale=False)
source = smpl_global_rotations(motion, flip_forward=True)
max_error = 0.0
max_wrist_error = 0.0
for frame in range(motion.frame_count):
    bpy.context.scene.frame_set(frame + 1)
    bpy.context.view_layer.update()
    evaluated = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
    for side, sign, indices in (("left", 1, (16, 18, 20)), ("right", -1, (17, 19, 21))):
        for joint, child, index in (("shoulder", "elbow", indices[0]),
                                    ("elbow", "wrist", indices[1])):
            bone = evaluated.pose.bones[mapping[f"{side}_{joint}"]]
            end = evaluated.pose.bones[mapping[f"{side}_{child}"]]
            actual = np.array((end.head - bone.head).normalized())
            expected = source[frame, index] @ np.array([sign, 0, 0])
            error = float(np.linalg.norm(actual - expected))
            max_error = max(max_error, error)
            np.testing.assert_allclose(actual, expected, atol=2e-5,
                                       err_msg=f"frame {frame + 1}: {side} {joint}")
        bone = evaluated.pose.bones[mapping[f"{side}_wrist"]]
        actual = np.array(bone.matrix.to_quaternion().to_matrix())
        rest_rotation = np.array(bone.bone.matrix_local.to_quaternion().to_matrix())
        expected = source[frame, indices[2]] @ np.array(corrections[f"{side}_wrist"].to_matrix())
        actual_delta = actual @ rest_rotation.T
        max_wrist_error = max(max_wrist_error, float(np.max(np.abs(actual_delta - expected))))
        np.testing.assert_allclose(actual_delta, expected, atol=2e-5,
                                   err_msg=f"frame {frame + 1}: {side} wrist")
for bone in rig.data.bones:
    np.testing.assert_array_equal(bone.matrix_local, rest[bone.name])
print("PASS: actual MMD shoulder/forearm directions and wrist orientation", json.dumps({
    "frames": motion.frame_count, "neutralized_helpers": result.neutralized_shoulder_helpers,
    "max_direction_error": max_error, "max_wrist_matrix_error": max_wrist_error,
}))
