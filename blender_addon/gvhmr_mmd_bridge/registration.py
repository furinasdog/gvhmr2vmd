"""Centralized Blender class registration."""

import bpy
from bpy.props import PointerProperty

from .operators import GVHMRMMD_OT_apply, GVHMRMMD_OT_restore_ik, GVHMRMMD_OT_validate
from .properties import GVHMRMMDProperties
from .ui import GVHMRMMD_PT_panel

CLASSES = (
    GVHMRMMDProperties,
    GVHMRMMD_OT_validate,
    GVHMRMMD_OT_apply,
    GVHMRMMD_OT_restore_ik,
    GVHMRMMD_PT_panel,
)


def register_addon():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gvhmr_mmd = PointerProperty(type=GVHMRMMDProperties)


def unregister_addon():
    if hasattr(bpy.types.Scene, "gvhmr_mmd"):
        del bpy.types.Scene.gvhmr_mmd
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
