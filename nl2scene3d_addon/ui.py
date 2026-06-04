# nl2scene3d/ui.py
"""
Interfaccia Blender: preferenze (solo il seed del randomizer), pannello in
sidebar, la lista degli override (fisso / padre / etichetta) e i comandi.

Override manuali:
  - Ogni voce e' un NL2_ObjectOverride (nome + 'fisso' + 'padre' + 'label').
  - 'label' decide se il nome di quell'oggetto compare nei render. E' attivo di
    default; spegnilo per togliere l'etichetta di quell'oggetto.

Questo modulo viene importato SOLO dentro register(): contiene bpy.
"""

import bpy  # type: ignore
from bpy.props import (  # type: ignore
    BoolProperty,
    CollectionProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Panel, PropertyGroup, UIList  # type: ignore


# ---------------------------------------------------------------------------
# Preferenze (solo il seed del randomizer)
# ---------------------------------------------------------------------------

class NL2SCENE3D_AddonPreferences(AddonPreferences):
    """Impostazioni dell'add-on."""

    bl_idname = __package__

    seed: IntProperty(  # type: ignore
        name="Randomizer Seed",
        description="Seme del disordine. 0 = casuale a ogni click; >0 = riproducibile "
                    "(utile per testare piu' LLM sulla stessa scena disordinata)",
        default=0, min=0,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "seed")
        layout.label(
            text="Flusso manuale: esporta il prompt, usalo nel tuo LLM, incolla qui la risposta.",
            icon="INFO",
        )


def get_prefs(context):
    """Ritorna le preferenze dell'add-on, o None se non disponibili."""
    addons = getattr(getattr(context, "preferences", None), "addons", None)
    if not addons:
        return None
    entry = addons.get(__package__)
    return entry.preferences if entry else None


# ---------------------------------------------------------------------------
# Override manuali: dato per-oggetto + lista
# ---------------------------------------------------------------------------

class NL2_ObjectOverride(PropertyGroup):
    """Una riga della lista: nome, fisso/mobile, padre, ed etichetta nei render."""

    name: StringProperty(name="Object")  # type: ignore
    fixed: BoolProperty(  # type: ignore
        name="Fisso",
        description="Se attivo, l'oggetto non verra' mai spostato",
        default=False,
    )
    parent: StringProperty(  # type: ignore
        name="Padre",
        description="Nome dell'oggetto padre: questo oggetto si muovera' insieme a lui "
                    "(vuoto = nessun padre)",
        default="",
    )
    label: BoolProperty(  # type: ignore
        name="Etichetta",
        description="Se attivo, il nome di questo oggetto compare nei render. "
                    "Spegnilo per non etichettarlo",
        default=True,
    )


class NL2SCENE3D_UL_overrides(UIList):
    """Lista oggetti: etichetta nei render + padre + lucchetto fisso/mobile."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            # toggle ETICHETTA (occhio): acceso = etichettato nei render
            row.prop(
                item, "label", text="", toggle=True,
                icon="HIDE_OFF" if item.label else "HIDE_ON",
            )
            row.label(text=item.name, icon="OBJECT_DATA")
            # Ricerca del padre tra gli oggetti della scena (vuoto = nessuno).
            row.prop_search(item, "parent", context.scene, "objects", text="", icon="CON_CHILDOF")
            row.prop(
                item, "fixed", text="", toggle=True,
                icon="LOCKED" if item.fixed else "UNLOCKED",
            )
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.name)


# ---------------------------------------------------------------------------
# Pannello in sidebar
# ---------------------------------------------------------------------------

class NL2SCENE3D_PT_main_panel(Panel):
    """Pannello principale, nella sidebar della 3D View."""

    bl_label       = "NL2Scene3D"
    bl_idname      = "NL2SCENE3D_PT_main_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "NL2Scene3D"

    def draw(self, context):
        layout = self.layout
        scene  = context.scene
        prefs  = get_prefs(context)

        box = layout.box()
        if prefs is None:
            box.label(text="Add-on non abilitato correttamente", icon="ERROR")
            return
        box.label(text="Flusso LLM manuale (esporta / applica)", icon="SHADERFX")

        layout.separator()
        layout.operator("nl2scene3d.inspect", text="Inspect Scene (dry-run)", icon="VIEWZOOM")
        layout.operator("nl2scene3d.scale_to_real", text="Scala a misura reale", icon="FULLSCREEN_ENTER")
        # --- Override manuali: etichetta / fisso / padre ---
        layout.separator()
        col = layout.column(align=True)
        col.prop(scene, "nl2_overrides_enabled", text="Override manuali (etichetta / fisso / padre)")
        if scene.nl2_overrides_enabled:
            row = col.row(align=True)
            row.operator("nl2scene3d.overrides_sync", text="Sincronizza", icon="FILE_REFRESH")
            row.operator("nl2scene3d.overrides_autodetect", text="Auto fisso", icon="ZOOM_SELECTED")
            row.operator("nl2scene3d.overrides_clear", text="", icon="TRASH")

            row2 = col.row(align=True)
            row2.operator("nl2scene3d.overrides_suggest_groups", text="Suggerisci gruppi", icon="CON_CHILDOF")
            op_on  = row2.operator("nl2scene3d.overrides_labels_all", text="", icon="HIDE_OFF")
            op_on.value = True
            op_off = row2.operator("nl2scene3d.overrides_labels_all", text="", icon="HIDE_ON")
            op_off.value = False

            if len(scene.nl2_overrides) == 0:
                col.label(text="Premi 'Sincronizza' per popolare la lista.", icon="INFO")
            else:
                col.template_list(
                    "NL2SCENE3D_UL_overrides", "",
                    scene, "nl2_overrides",
                    scene, "nl2_overrides_index",
                    rows=6,
                )
                col.label(text="Occhio = etichetta render, Icona catena = padre, lucchetto = fisso/mobile.", icon="INFO")

        # --- Step 1: disordina + reset ---
        layout.separator()
        col1 = layout.column(align=True)
        col1.label(text="Step 1: Disordina (opzionale)")
        col1.operator("nl2scene3d.randomize", text="Randomize Layout", icon="RECOVER_LAST")
        col1.operator("nl2scene3d.reset_home", text="Reset to Original", icon="LOOP_BACK")
        if scene.nl2_has_home:
            col1.label(text="Originale salvato.", icon="CHECKMARK")
        else:
            col1.label(text="Nessun originale (premi Randomize una volta).", icon="INFO")

        # --- Step 2: riordino con AI (flusso manuale) ---
        layout.separator()
        col2 = layout.column(align=True)
        col2.label(text="Step 2: Riordina con AI (manuale)")
        col2.operator("nl2scene3d.render_labeled", text="0. Render con etichette", icon="RENDER_STILL")
        col2.operator("nl2scene3d.export_for_llm", text="1. Esporta prompt per LLM", icon="EXPORT")
        col2.label(text="Allega i 2 render + copia 'NL2_AI_Prompt' nel tuo LLM,", icon="INFO")
        col2.label(text="incolla la risposta nel Text 'NL2_AI_Response', poi:")
        row3 = col2.row(align=True)
        row3.operator("nl2scene3d.apply_from_text", text="2. Applica (incollato)", icon="PASTEDOWN")
        row3.operator("nl2scene3d.apply_from_file", text="…da file", icon="FILEBROWSER")


# ---------------------------------------------------------------------------
# Registrazione
# ---------------------------------------------------------------------------

_classes = (
    NL2SCENE3D_AddonPreferences,
    NL2_ObjectOverride,
    NL2SCENE3D_UL_overrides,
    NL2SCENE3D_PT_main_panel,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.nl2_overrides = CollectionProperty(type=NL2_ObjectOverride)
    bpy.types.Scene.nl2_overrides_index = IntProperty(default=0)
    bpy.types.Scene.nl2_overrides_enabled = BoolProperty(
        name="Override manuali",
        description="Se attivo, usa le scelte manuali (etichetta / fisso / padre) "
                    "al posto dell'automatico",
        default=False,
    )
    bpy.types.Scene.nl2_has_home = BoolProperty(
        name="Ha stato originale",
        description="Indica se e' stato salvato uno stato originale per il reset",
        default=False,
    )


def unregister():
    for attr in ("nl2_overrides", "nl2_overrides_index", "nl2_overrides_enabled",
                 "nl2_has_home"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)

    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass