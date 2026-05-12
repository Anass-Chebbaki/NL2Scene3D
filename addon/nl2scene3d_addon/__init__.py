# addon/nl2scene3d_addon/__init__.py
"""
NL2Scene3D Blender Add-on.

Scene reorganization from random to ordered via Multimodal Language Models.
"""

bl_info = {
    "name": "NL2Scene3D",
    "author": "NL2Scene3D Team",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > NL2Scene3D",
    "description": "Reorganize 3D scenes via Multimodal Language Models",
    "category": "3D View",
}

import bpy
from bpy.types import Panel, Operator


# ----------------------------------------------------------------------
# Operator: a simple hello world action to verify registration works
# ----------------------------------------------------------------------
class NL2SCENE3D_OT_hello(Operator):
    """Test operator to verify the add-on is loaded correctly."""
    bl_idname = "nl2scene3d.hello"
    bl_label = "Say Hello"
    bl_description = "Print a hello message to confirm the add-on works"

    def execute(self, context):
        self.report({'INFO'}, "Hello from NL2Scene3D add-on!")
        print("[NL2Scene3D] Hello from the add-on!")
        return {'FINISHED'}


# ----------------------------------------------------------------------
# Panel: appears in the 3D View sidebar (press N to toggle)
# ----------------------------------------------------------------------
class NL2SCENE3D_PT_main_panel(Panel):
    """Main panel of the NL2Scene3D add-on."""
    bl_label = "NL2Scene3D"
    bl_idname = "NL2SCENE3D_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NL2Scene3D"

    def draw(self, context):
        layout = self.layout

        layout.label(text="Scene Reorganization", icon='SCENE_DATA')

        layout.separator()

        layout.label(text="Add-on is working!", icon='CHECKMARK')

        layout.separator()

        layout.operator("nl2scene3d.hello", text="Say Hello", icon='INFO')


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
classes = (
    NL2SCENE3D_OT_hello,
    NL2SCENE3D_PT_main_panel,
)


def register():
    """Register all add-on classes with Blender."""
    for cls in classes:
        bpy.utils.register_class(cls)
    print("[NL2Scene3D] Add-on registered.")


def unregister():
    """Unregister all add-on classes from Blender."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("[NL2Scene3D] Add-on unregistered.")


if __name__ == "__main__":
    register()