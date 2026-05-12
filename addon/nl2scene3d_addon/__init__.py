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

import sys
from pathlib import Path

# Set up sys.path BEFORE any other import:
# - Add the add-on directory so 'core' is importable.
# - Add the vendor/ subdirectory so bundled external libraries
#   (dotenv, google-genai, Pillow, etc.) are available inside Blender.
_ADDON_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _ADDON_DIR / "vendor"

for path in (_ADDON_DIR, _VENDOR_DIR):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

import bpy
from bpy.types import Panel, Operator


# ----------------------------------------------------------------------
# Operator: inspect the current Blender scene using core modules
# ----------------------------------------------------------------------
class NL2SCENE3D_OT_inspect_scene(Operator):
    """Inspect the current scene using core pipeline modules."""
    bl_idname = "nl2scene3d.inspect_scene"
    bl_label = "Inspect Scene"
    bl_description = "Analyze current scene objects using the core pipeline modules"

    def execute(self, context):
        try:
            from core.scene_loader import SceneLoader
            from core.config import get_config

            config = get_config()
            loader = SceneLoader(config.pipeline)
            state = loader.extract_scene_state(scene_name="inspected_scene")

            total = len(state.objects)
            movable = sum(1 for obj in state.objects if obj.is_movable)
            bounds = state.room_bounds

            msg = (
                f"Scene has {total} objects ({movable} movable). "
                f"Bounds: X[{bounds.x_min:.2f}, {bounds.x_max:.2f}] "
                f"Y[{bounds.y_min:.2f}, {bounds.y_max:.2f}]"
            )

            self.report({'INFO'}, msg)
            print(f"[NL2Scene3D] {msg}")

            for obj in state.objects:
                print(
                    f"[NL2Scene3D]   {obj.name:30s} "
                    f"category={obj.category:20s} movable={obj.is_movable}"
                )

            return {'FINISHED'}

        except Exception as exc:
            error_msg = f"Inspection failed: {exc}"
            self.report({'ERROR'}, error_msg)
            print(f"[NL2Scene3D] {error_msg}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


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

        box = layout.box()
        box.label(text="Scene Analysis", icon='VIEWZOOM')
        box.operator(
            "nl2scene3d.inspect_scene",
            text="Inspect Scene",
            icon='OUTLINER_OB_GROUP_INSTANCE',
        )


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
classes = (
    NL2SCENE3D_OT_inspect_scene,
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