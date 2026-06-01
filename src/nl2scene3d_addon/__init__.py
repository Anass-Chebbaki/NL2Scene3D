# nl2scene3d_addon/__init__.py
"""
NL2Scene3D Blender Add-on.

Integrates the NL2Scene3D pipeline into Blender, exposing:
  - Scene randomization (controlled disorganization)
  - Scene reorganization via Google Gemini (multimodal LLM)
  - API configuration through Blender Addon Preferences
"""

import os
import sys
import traceback
from pathlib import Path

import bpy                                                       # type: ignore
from bpy.props import EnumProperty, StringProperty              # type: ignore
from bpy.types import AddonPreferences, Operator, Panel         # type: ignore


# ---------------------------------------------------------------------------
# Addon metadata
# ---------------------------------------------------------------------------

bl_info = {
    "name":        "NL2Scene3D",
    "author":      "NL2Scene3D Team",
    "version":     (0, 1, 0),
    "blender":     (5, 1, 0),
    "location":    "View3D > Sidebar > NL2Scene3D",
    "description": "Reorganize 3D scenes via Multimodal Language Models (Gemini)",
    "category":    "3D View",
}


# ---------------------------------------------------------------------------
# Path setup — make bundled packages discoverable
# ---------------------------------------------------------------------------

_ADDON_DIR  = Path(__file__).resolve().parent
_VENDOR_DIR = _ADDON_DIR / "vendor"

print(f"[NL2Scene3D] Addon dir  : {_ADDON_DIR}")
print(f"[NL2Scene3D] Vendor dir : {_VENDOR_DIR}  (exists={_VENDOR_DIR.exists()})")
print(f"[NL2Scene3D] dotenv dir : {(_VENDOR_DIR / 'dotenv').exists()}")

if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

if str(_ADDON_DIR) not in sys.path:
    sys.path.insert(0, str(_ADDON_DIR))

# Force-remove 'dotenv' from sys.modules if it is incomplete or shadowed from an external folder.
# This prevents namespace conflicts in Blender with other addons that use dummy or incompatible dotenv packages.
if "dotenv" in sys.modules:
    dot_mod = sys.modules["dotenv"]
    if (
        not dot_mod
        or not hasattr(dot_mod, "load_dotenv")
        or (hasattr(dot_mod, "__file__") and dot_mod.__file__ and "vendor" not in dot_mod.__file__)
    ):
        sys.modules.pop("dotenv", None)

try:
    from dotenv import load_dotenv
    print(f"[NL2Scene3D] dotenv loaded from: {load_dotenv.__module__}")
except ImportError as exc:
    print(f"[NL2Scene3D] dotenv import failed: {exc}")


# ---------------------------------------------------------------------------
# Addon preferences
# ---------------------------------------------------------------------------

class NL2SCENE3D_AddonPreferences(AddonPreferences):
    """Backend choice + Gemini and Ollama settings."""

    bl_idname = Path(__file__).resolve().parent.name

    backend: EnumProperty(             # type: ignore
        name="Backend",
        description="Where to run the model",
        items=[
            ("GEMINI", "Gemini (cloud)", "Google Gemini API (needs a key)"),
            ("OLLAMA", "Ollama (local)", "Local model via Ollama (no key, no quota)"),
        ],
        default="GEMINI",
    )

    api_key: StringProperty(           # type: ignore
        name="Gemini API Key", description="Your Google Gemini API key",
        default="", subtype="PASSWORD",
    )

    model_name: EnumProperty(          # type: ignore
        name="Gemini Model", description="Gemini model to use",
        items=[
            ("gemini-2.5-flash", "Gemini 2.5 Flash", "Stable flash model"),
            ("gemini-3.5-flash", "Gemini 3.5 Flash", "Latest high-speed model"),
        ],
        default="gemini-2.5-flash",
    )

    ollama_model: StringProperty(      # type: ignore
        name="Ollama Model", description="Local model tag (e.g. qwen3.5:4b, qwen3.5:2b)",
        default="qwen3.5:4b",
    )

    ollama_url: StringProperty(        # type: ignore
        name="Ollama URL", description="Ollama server URL",
        default="http://localhost:11434",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "backend")
        if self.backend == "GEMINI":
            layout.prop(self, "api_key")
            layout.prop(self, "model_name")
            if not self.api_key:
                layout.label(text="Get an API key at aistudio.google.com", icon="INFO")
        else:
            layout.prop(self, "ollama_model")
            layout.prop(self, "ollama_url")
            layout.label(text="Make sure the Ollama app is running.", icon="INFO")


# ---------------------------------------------------------------------------
# Pipeline context factory
# ---------------------------------------------------------------------------

def get_pipeline_context():
    """
    Builds the core pipeline components from the current addon preferences.
    Returns (config, loader, applicator, randomizer, reorganizer) or 5x None.
    """
    addon_id = Path(__file__).resolve().parent.name

    if not (bpy.context and bpy.context.preferences and bpy.context.preferences.addons):
        print("[NL2Scene3D] Blender context or preferences not available.")
        return None, None, None, None, None

    addon_ref = bpy.context.preferences.addons.get(addon_id)  # type: ignore
    if addon_ref is None:
        print(f"[NL2Scene3D] Preferences for '{addon_id}' not found.")
        return None, None, None, None, None
    prefs = addon_ref.preferences

    backend = getattr(prefs, "backend", "GEMINI")

    # config validation requires a non-empty GEMINI_API_KEY even in Ollama mode:
    # set a harmless placeholder (the Ollama client never uses it).
    if backend == "OLLAMA":
        os.environ.setdefault("GEMINI_API_KEY", "ollama-local")
    elif prefs.api_key:
        os.environ["GEMINI_API_KEY"] = prefs.api_key

    from nl2scene3d.config           import get_config
    from nl2scene3d.gemini_client    import GeminiClient
    from nl2scene3d.randomizer       import SceneRandomizer
    from nl2scene3d.scene_applicator import SceneApplicator
    from nl2scene3d.scene_state      import SceneLoader
    from nl2scene3d.scene_reorganizer import SceneReorganizer
    import nl2scene3d

    config = get_config()

    if backend == "OLLAMA":
        from nl2scene3d.ollama_client import OllamaClient
        client = OllamaClient(
            model=prefs.ollama_model,
            base_url=prefs.ollama_url,
            temperature=config.gemini.temperature,
        )
    else:
        config.gemini.model_primary = prefs.model_name  # UI override
        client = GeminiClient(config.gemini)

    loader      = SceneLoader(config.pipeline)
    applicator  = SceneApplicator()
    randomizer  = SceneRandomizer(config.randomizer)
    prompts_dir = Path(nl2scene3d.__file__).parent / "config" / "prompts"
    reorganizer = SceneReorganizer(client, prompts_dir)

    return config, loader, applicator, randomizer, reorganizer


# ---------------------------------------------------------------------------
# Operator: Randomize Scene
# ---------------------------------------------------------------------------

class NL2SCENE3D_OT_randomize(Operator):
    """Randomly scatters movable objects within the room bounds."""

    bl_idname  = "nl2scene3d.randomize"
    bl_label   = "Randomize Scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            config, loader, applicator, randomizer, _ = get_pipeline_context()

            if config is None:
                self.report({"ERROR"}, "Add-on configuration not found. Check Preferences.")
                return {"CANCELLED"}

            if not (loader and applicator and randomizer):
                self.report({"ERROR"}, "Add-on components not initialized properly.")
                return {"CANCELLED"}

            wm = context.window_manager
            wm.progress_begin(0, 100)
            for window in wm.windows:
                window.cursor_set("WAIT")

            wm.progress_update(20)
            self.report({"INFO"}, "Extracting scene state...")
            state = loader.extract_scene_state()

            wm.progress_update(50)
            self.report({"INFO"}, "Randomizing layout...")
            randomized_state = randomizer.randomize(state)

            wm.progress_update(80)
            self.report({"INFO"}, "Applying randomized state...")
            applicator.apply_state(randomized_state)

            wm.progress_update(100)
            wm.progress_end()
            for window in wm.windows:
                window.cursor_set("DEFAULT")

            count = len(state.movable_objects)
            self.report({"INFO"}, f"Randomized {count} objects.")
            return {"FINISHED"}

        except Exception as exc:
            self._reset_ui(context)
            self.report({"ERROR"}, f"Randomization failed: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}

    @staticmethod
    def _reset_ui(context):
        try:
            wm = context.window_manager
            wm.progress_end()
            for window in wm.windows:
                window.cursor_set("DEFAULT")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Operator: Reorganize Scene (AI)
# ---------------------------------------------------------------------------

class NL2SCENE3D_OT_reorganize(Operator):
    """Reorganizes the scene using the Gemini multimodal LLM pipeline."""

    bl_idname  = "nl2scene3d.reorganize"
    bl_label   = "Reorganize Scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        import tempfile
        try:
            config, loader, applicator, _, reorganizer = get_pipeline_context()
            if config is None:
                self.report({"ERROR"}, "Add-on configuration not found. Check Preferences.")
                return {"CANCELLED"}
            if not (loader and applicator and reorganizer):
                self.report({"ERROR"}, "Add-on components not initialized properly.")
                return {"CANCELLED"}

            from nl2scene3d.blender.renderer import BlenderRenderer

            wm = context.window_manager
            wm.progress_begin(0, 100)
            for window in wm.windows:
                window.cursor_set("WAIT")

            # 1. Extract current (disordered) state.
            wm.progress_update(10)
            self.report({"INFO"}, "Extracting current scene state...")
            state = loader.extract_scene_state()

            # 2. Render reference views (top + isometric).
            wm.progress_update(30)
            self.report({"INFO"}, "Rendering reference views (top + iso)...")
            out_dir = Path(tempfile.gettempdir()) / "nl2scene3d_reorder_ref"
            renderer = BlenderRenderer(output_dir=out_dir, config=config.render)
            render_paths = renderer.render_step(
                step_name="reorder_ref", state=state, quality="preview"
            )
            image_paths = [render_paths["top"], render_paths["iso"]]

            # 3. Multimodal reorder (images + JSON).
            wm.progress_update(60)
            self.report({"INFO"}, "Calling Gemini (images + JSON)... (20-40s)")
            new_state = reorganizer.reorganize_multimodal(state, image_paths)

            if new_state.pipeline_step == "reordered_failed":
                wm.progress_end()
                for window in wm.windows:
                    window.cursor_set("DEFAULT")
                self.report({"ERROR"}, "LLM response could not be parsed. Scene unchanged.")
                return {"CANCELLED"}

            # 4. Apply the new layout.
            wm.progress_update(85)
            self.report({"INFO"}, "Applying reorganized layout...")
            applicator.apply_state(new_state)

            wm.progress_update(100)
            wm.progress_end()
            for window in wm.windows:
                window.cursor_set("DEFAULT")

            clamped = new_state.metadata.get("clamped_count", 0)
            missing = new_state.metadata.get("missing_count", 0)
            self.report(
                {"INFO"},
                f"Multimodal reorder done. Clamped: {clamped}, missing: {missing}.",
            )
            return {"FINISHED"}

        except Exception as exc:
            try:
                wm = context.window_manager
                wm.progress_end()
                for window in wm.windows:
                    window.cursor_set("DEFAULT")
            except Exception:
                pass
            self.report({"ERROR"}, f"Reorder failed: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}

    @staticmethod
    def _reset_ui(context):
        try:
            wm = context.window_manager
            wm.progress_end()
            for window in wm.windows:
                window.cursor_set("DEFAULT")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Panel: sidebar UI
# ---------------------------------------------------------------------------

class NL2SCENE3D_PT_main_panel(Panel):
    """Main NL2Scene3D panel, shown in the 3D View sidebar."""

    bl_label       = "NL2Scene3D"
    bl_idname      = "NL2SCENE3D_PT_main_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "NL2Scene3D"

    def draw(self, context):
        layout = self.layout
        if layout is None:
            return

        addon_id = Path(__file__).resolve().parent.name

        try:
            prefs_attr  = getattr(context, "preferences", None)
            addons_attr = getattr(prefs_attr, "addons", None) if prefs_attr else None
            addon_ref   = addons_attr.get(addon_id) if addons_attr else None

            if addon_ref is None:
                layout.label(text="Add-on not enabled properly", icon="ERROR")
                return

            prefs = addon_ref.preferences

            # Header
            layout.label(text="Smart Reorganization", icon="OUTLINER_OB_GROUP_INSTANCE")

            # API key status box
            box = layout.box()
            if getattr(prefs, "backend", "GEMINI") == "OLLAMA":
                box.label(text=f"Local: {prefs.ollama_model}", icon="CHECKMARK")
            elif not prefs.api_key:
                box.label(text="API Key Missing!", icon="ERROR")
                box.operator("wm.url_open", text="Get Key").url = "https://aistudio.google.com/"
            else:
                box.label(text=f"Gemini: {prefs.model_name}", icon="CHECKMARK")

            layout.separator()
            layout.label(text="Scene Controls", icon="SCENE_DATA")

            # Step 1
            col1 = layout.column(align=True)
            col1.label(text="Step 1: Disorganize (Optional)")
            col1.operator("nl2scene3d.randomize", text="Randomize Layout", icon="RECOVER_LAST")

            layout.separator()

            # Step 2
            col2 = layout.column(align=True)
            col2.label(text="Step 2: Reorganize")
            col2.operator("nl2scene3d.reorganize", text="AI Reorder", icon="PLAY")
            col2.scale_y = 1.4

        except Exception as exc:
            layout.label(text=f"Error: {str(exc)[:30]}...", icon="ERROR")
            print(f"[NL2Scene3D] UI error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    NL2SCENE3D_AddonPreferences,
    NL2SCENE3D_OT_randomize,
    NL2SCENE3D_OT_reorganize,
    NL2SCENE3D_PT_main_panel,
)


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as exc:
            print(f"[NL2Scene3D] Registration error for {cls.__name__}: {exc}")


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()