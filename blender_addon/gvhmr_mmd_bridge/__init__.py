"""GVHMR to MMD Blender add-on."""

bl_info = {
    "name": "GVHMR to MMD Bridge",
    "author": "GVHMR-to-MMD contributors",
    "version": (0, 2, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > GVHMR MMD",
    "description": "Retarget safe GVHMR NPZ motion files to MMD armatures",
    "category": "Animation",
    "license": "GPL-3.0-or-later",
}


def register():
    from .registration import register_addon

    register_addon()


def unregister():
    from .registration import unregister_addon

    unregister_addon()


if __name__ == "__main__":
    register()
