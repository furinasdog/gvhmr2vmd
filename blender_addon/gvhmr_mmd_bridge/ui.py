"""3D View sidebar panel."""

import bpy


class GVHMRMMD_PT_panel(bpy.types.Panel):
    bl_label = "GVHMR → MMD"
    bl_idname = "GVHMRMMD_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GVHMR MMD"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gvhmr_mmd

        layout.prop(props, "motion_path")
        layout.prop(props, "armature")
        layout.operator("gvhmr_mmd.validate", icon="CHECKMARK")

        box = layout.box()
        box.label(text="重定向设置")
        box.prop(props, "start_frame")
        box.prop(props, "sync_fps")
        box.prop(props, "auto_scale")
        if not props.auto_scale:
            box.prop(props, "manual_scale")
        box.prop(props, "flip_forward")
        box.prop(props, "compensate_arm_rest_pose")
        box.prop(props, "apply_hands")
        if props.apply_hands:
            box.prop(props, "limit_finger_splay")
            if props.limit_finger_splay:
                box.prop(props, "limit_finger_angles")
                if props.limit_finger_angles:
                    box.prop(props, "finger_max_flexion")
                    box.prop(props, "finger_max_extension")
            box.prop(props, "hand_confidence_threshold")
        box.prop(props, "disable_ik")

        layout.operator("gvhmr_mmd.apply", icon="ACTION")
        layout.operator("gvhmr_mmd.restore_ik", icon="CONSTRAINT_BONE")

        if props.status:
            status_box = layout.box()
            status_box.label(text=props.status, icon="INFO")
