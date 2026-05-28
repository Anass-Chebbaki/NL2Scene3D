# addon/nl2scene3d_addon/__init__.py
"""
NL2Scene3D Blender Add-on.

Professional integration of the NL2Scene3D pipeline:
- Randomize scene (disorganize)
- Reorganize scene via Multimodal Language Models (Gemini)
- API configuration via Addon Preferences
"""

import sys
import os
import traceback
from pathlib import Path

# Addon metadata for Blender
bl_info = {
    "name": "NL2Scene3D",
    "author": "NL2Scene3D Team",
    "version": (0, 1, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > NL2Scene3D",
    "description": "Reorganize 3D scenes via Multimodal Language Models (Gemini)",
    "category": "3D View",
}

# ----------------------------------------------------------------------
# PATH SETUP: Ensure bundled packages and vendor libs are discoverable
# ----------------------------------------------------------------------
_ADDON_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _ADDON_DIR / "vendor"

print(f"[NL2Scene3D DEBUG] _ADDON_DIR = {_ADDON_DIR}")
print(f"[NL2Scene3D DEBUG] _VENDOR_DIR = {_VENDOR_DIR}")
print(f"[NL2Scene3D DEBUG] _VENDOR_DIR exists = {_VENDOR_DIR.exists()}")
print(f"[NL2Scene3D DEBUG] dotenv folder exists = {(_VENDOR_DIR / 'dotenv').exists()}")

# Add vendor directory to sys.path
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

if str(_ADDON_DIR) not in sys.path:
    sys.path.insert(0, str(_ADDON_DIR))

# Verifica che dotenv sia importabile
try:
    from dotenv import load_dotenv
    print(f"[NL2Scene3D DEBUG] dotenv import OK, location: {load_dotenv.__module__}")
except ImportError as e:
    print(f"[NL2Scene3D DEBUG] dotenv import FAILED: {e}")


import bpy  # type: ignore
from bpy.types import Panel, Operator, AddonPreferences  # type: ignore
from bpy.props import StringProperty, EnumProperty  # type: ignore

# ----------------------------------------------------------------------
# PREFERENCES: Store API keys and user settings
# ----------------------------------------------------------------------
class NL2SCENE3D_AddonPreferences(AddonPreferences):
    bl_idname = __package__ or "nl2scene3d_addon"

    api_key: StringProperty( # type: ignore
        name="Gemini API Key",
        description="Enter your Google Gemini API Key",
        default="",
        subtype='PASSWORD',
    )

    model_name: EnumProperty( # type: ignore
        name="Model",
        description="Choose the Gemini model to use",
        items=[
            ('gemini-3.5-flash', "Gemini 3.5 Flash", "Latest high-speed model"),
            ('gemini-2.5-flash', "Gemini 2.5 Flash", "Standard stable flash model"),
            ('gemini-2.5-pro', "Gemini 2.5 Pro", "High intelligence model"),
        ],
        default='gemini-3.5-flash',
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "api_key")
        layout.prop(self, "model_name")
        
        if not self.api_key:
            layout.label(text="Get an API key at aistudio.google.com", icon='INFO')


# ----------------------------------------------------------------------
# HELPER: Get core components with correct config
# ----------------------------------------------------------------------
def get_pipeline_context():
    """Initialize core components using current addon settings."""
    addon_id = __package__ or "nl2scene3d_addon"
    if not (bpy.context and bpy.context.preferences and bpy.context.preferences.addons):
        print("[NL2Scene3D] Preferences or context not available.")
        return None, None, None, None, None
    try:
        prefs_attr = getattr(bpy.context, "preferences", None)
        addons_attr = getattr(prefs_attr, "addons", None) if prefs_attr else None
        addon_ref = addons_attr.get(addon_id) if addons_attr else None
        if not addon_ref:
            raise KeyError()
        prefs = addon_ref.preferences
    except (KeyError, AttributeError):
        print(f"[NL2Scene3D] Errore: preferenze per '{addon_id}' non trovate.")
        return None, None, None, None, None
    
    # Set environment variables BEFORE importing core modules that validate them
    if prefs.api_key:
        os.environ["GEMINI_API_KEY"] = prefs.api_key
    
    from nl2scene3d.config import get_config, reset_config
    from nl2scene3d.gemini_client import GeminiClient
    from nl2scene3d.scene_loader import SceneLoader
    from nl2scene3d.scene_applicator import SceneApplicator
    from nl2scene3d.randomizer import SceneRandomizer
    from nl2scene3d.scene_reorganizer import SceneReorganizer

    reset_config()
    config = get_config()

    # Apply UI overrides to config
    config.gemini.model_primary = prefs.model_name

    client = GeminiClient(config.gemini)
    loader = SceneLoader(config.pipeline)
    applicator = SceneApplicator()

    # config.randomizer è già un RandomizerConfig completo — usato direttamente
    randomizer = SceneRandomizer(config.randomizer)
    
    import nl2scene3d
    # Nuovo percorso: i prompt sono dentro config/prompts nel pacchetto
    prompts_dir = Path(nl2scene3d.__file__).parent / "config" / "prompts"
    reorganizer = SceneReorganizer(client, prompts_dir)
    
    return config, loader, applicator, randomizer, reorganizer


# ----------------------------------------------------------------------
# OPERATOR: Randomize Scene (Disorganize)
# ----------------------------------------------------------------------
class NL2SCENE3D_OT_randomize(Operator):
    """Randomly scatter movable objects within room bounds."""
    bl_idname = "nl2scene3d.randomize"
    bl_label = "Randomize Scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            config_data = get_pipeline_context()
            if not config_data or not config_data[0]:
                self.report({'ERROR'}, "Add-on configuration not found. Check Preferences.")
                return {'CANCELLED'}

            config, loader, applicator, randomizer, reorganizer = config_data
            if not (loader and applicator and randomizer):
                self.report({'ERROR'}, "Add-on components not initialized properly.")
                return {'CANCELLED'}

            wm = context.window_manager
            wm.progress_begin(0, 100)
            for window in wm.windows:
                window.cursor_set('WAIT')

            wm.progress_update(20)
            self.report({'INFO'}, "Extracting scene state...")
            print("[NL2Scene3D] Extracting scene state...")
            state = loader.extract_scene_state()

            wm.progress_update(50)
            self.report({'INFO'}, "Randomizing layout...")
            print("[NL2Scene3D] Randomizing layout...")
            randomized_state = randomizer.randomize(state)

            wm.progress_update(80)
            self.report({'INFO'}, "Applying randomized state...")
            print("[NL2Scene3D] Applying randomized state...")
            applicator.apply_state(randomized_state)

            wm.progress_update(100)
            wm.progress_end()
            for window in wm.windows:
                window.cursor_set('DEFAULT')

            count = len(state.movable_objects)
            msg = f"Randomized {count} objects."
            self.report({'INFO'}, msg)
            print(f"[NL2Scene3D] {msg}")
            return {'FINISHED'}

        except Exception as e:
            try:
                wm = context.window_manager
                wm.progress_end()
                for window in wm.windows:
                    window.cursor_set('DEFAULT')
            except Exception:
                pass

            self.report({'ERROR'}, f"Randomization failed: {e}")
            traceback.print_exc()
            return {'CANCELLED'}


# ----------------------------------------------------------------------
# OPERATOR: Reorganize Scene (Core AI Feature)
# ----------------------------------------------------------------------
class NL2SCENE3D_OT_reorganize(Operator):
    """Reorganize the scene using Multimodal LLM logic."""
    bl_idname = "nl2scene3d.reorganize"
    bl_label = "Reorganize Scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            config_data = get_pipeline_context()
            if not config_data or not config_data[0]:
                self.report({'ERROR'}, "Add-on configuration not found. Check Preferences.")
                return {'CANCELLED'}

            config, loader, applicator, _, reorganizer = config_data
            if not (loader and applicator and reorganizer):
                self.report({'ERROR'}, "Add-on components not initialized properly.")
                return {'CANCELLED'}

            # ----------------------------------------------------------
            # Feedback visivo: cursore "wait"
            # ----------------------------------------------------------
            wm = context.window_manager
            wm.progress_begin(0, 100)
            for window in wm.windows:
                window.cursor_set('WAIT')

            # ----------------------------------------------------------
            # Step 1: Estrazione stato scena
            # ----------------------------------------------------------
            wm.progress_update(20)
            self.report({'INFO'}, "Extracting scene state...")
            print("[NL2Scene3D] Extracting scene state...")
            state = loader.extract_scene_state()

            root_count = len(state.root_movable_objects)
            print(f"[NL2Scene3D] Root movable objects: {root_count}")

            # ----------------------------------------------------------
            # Step 2: Chiamata Gemini text-only (solo oggetti padre)
            # ----------------------------------------------------------
            wm.progress_update(40)
            self.report({'INFO'}, f"Calling Gemini AI for {root_count} root objects... (10-30s)")
            print("[NL2Scene3D] Calling Gemini AI (text-only, flat JSON)...")

            new_state = reorganizer.reorganize(state)

            # ----------------------------------------------------------
            # Step 3: Applicazione risultato
            # ----------------------------------------------------------
            wm.progress_update(85)
            self.report({'INFO'}, "Applying reorganized state...")
            print("[NL2Scene3D] Applying reorganized state...")
            applicator.apply_state(new_state)

            # ----------------------------------------------------------
            # Cleanup feedback
            # ----------------------------------------------------------
            wm.progress_update(100)
            wm.progress_end()
            for window in wm.windows:
                window.cursor_set('DEFAULT')

            clamped = new_state.metadata.get('clamped_count', 0)
            missing = new_state.metadata.get('missing_count', 0)

            if new_state.pipeline_step == "reordered_failed":
                error = new_state.metadata.get('error', 'unknown')
                msg = f"Reorganization failed: {error}"
                self.report({'WARNING'}, msg)
                print(f"[NL2Scene3D] {msg}")
            else:
                msg = (
                    f"Reorganization complete. "
                    f"Clamped: {clamped}, Missing: {missing}."
                )
                self.report({'INFO'}, msg)
                print(f"[NL2Scene3D] {msg}")

            return {'FINISHED'}

        except Exception as e:
            try:
                wm = context.window_manager
                wm.progress_end()
                for window in wm.windows:
                    window.cursor_set('DEFAULT')
            except Exception:
                pass

            self.report({'ERROR'}, f"Reorganization failed: {e}")
            traceback.print_exc()
            return {'CANCELLED'}


# ----------------------------------------------------------------------
# PANEL: Main UI in Sidebar
# ----------------------------------------------------------------------
class NL2SCENE3D_PT_main_panel(Panel):
    bl_label = "NL2Scene3D"
    bl_idname = "NL2SCENE3D_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NL2Scene3D"

    def draw(self, context):
        layout = self.layout
        if layout is None:
            return
        addon_id = __package__ or "nl2scene3d_addon"
        
        try:
            # Recupero preferenze
            addon_ref = None
            prefs_attr = getattr(context, "preferences", None)
            addons_attr = getattr(prefs_attr, "addons", None) if prefs_attr else None
            if addons_attr:
                addon_ref = addons_attr.get(addon_id)
            
            if addon_ref is None:
                layout.label(text="Add-on not enabled properly", icon='ERROR')
                return
                
            prefs = addon_ref.preferences
            
            # Header
            layout.label(text="Smart Reorganization", icon='OUTLINER_OB_GROUP_INSTANCE')
            
            # Status Box
            box = layout.box()
            if not prefs.api_key:
                box.label(text="API Key Missing!", icon='ERROR')
                box.operator("wm.url_open", text="Get Key").url = "https://aistudio.google.com/"
            else:
                box.label(text=f"Model: {prefs.model_name}", icon='CHECKMARK')
            
            layout.separator()
            
            # Controls Header
            layout.label(text="Scene Controls", icon='SCENE_DATA')
            
            # Step 1
            col1 = layout.column(align=True)
            col1.label(text="Step 1: Disorganize (Optional)")
            col1.operator("nl2scene3d.randomize", text="Randomize Layout", icon='RECOVER_LAST')
            
            layout.separator()
            
            # Step 2
            col2 = layout.column(align=True)
            col2.label(text="Step 2: Reorganize")
            # Usiamo un'icona super sicura 'PLAY' invece di 'MOD_SIMULATE'
            col2.operator("nl2scene3d.reorganize", text="AI Reorder", icon='PLAY')
            col2.scale_y = 1.4
            
        except Exception as e:
            # Mostriamo l'errore reale nel pannello
            err_msg = str(e)
            layout.label(text=f"Error: {err_msg[:30]}...", icon='ERROR')
            print(f"[NL2Scene3D] UI Error: {traceback.format_exc()}")


# ----------------------------------------------------------------------
# REGISTRATION
# ----------------------------------------------------------------------
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
        except Exception as e:
            print(f"[NL2Scene3D] Registration error for {cls}: {e}")

def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            pass

if __name__ == "__main__":
    register()