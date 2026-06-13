# nl2scene3d/ui.py
"""
Interfaccia utente Blender dell'add-on NL2Scene3D.

Contiene:
    - NL2SCENE3D_AddonPreferences: preferenze dell'add-on (seed del randomizer).
    - NL2_ObjectOverride:          PropertyGroup per ogni voce della lista override.
    - NL2SCENE3D_UL_overrides:     UIList che mostra etichetta / padre / fisso.
    - NL2SCENE3D_PT_main_panel:    Pannello principale nella sidebar della 3D View.

Override manuali:
    Ogni voce e' un NL2_ObjectOverride con i campi:
        - name:       nome dell'oggetto Blender.
        - fixed:      se True l'oggetto non viene mai spostato.
        - parent:     nome dell'oggetto padre (vuoto = nessun padre).
        - label:      se True il nome compare nei render etichettati.
        - keep_scale: se True l'operatore 'Scala a misura reale' non tocca questo oggetto.

Questo modulo viene importato SOLO dentro register() perche' dipende da bpy.
"""

import bpy  # type: ignore
from bpy.props import (  # type: ignore
    BoolProperty,
    CollectionProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Panel, PropertyGroup, UIList  # type: ignore

# Pillow detection a livello modulo, non in ogni frame del draw().
# Blender chiama draw() per ogni aggiornamento UI; un import-attempt per frame
# e' inutile e leggermente costoso. Il flag viene calcolato una sola volta.
try:
    from PIL import Image as _PIL_Image  # noqa: F401
    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Preferenze dell'add-on
# ---------------------------------------------------------------------------

class NL2SCENE3D_AddonPreferences(AddonPreferences):
    """Impostazioni persistenti dell'add-on, accessibili da Edit > Preferences."""

    bl_idname = __package__

    seed: IntProperty(  # type: ignore
        name="Randomizer Seed",
        description=(
            "Seme del disordine. "
            "0 = casuale a ogni click; "
            "valore > 0 = riproducibile (utile per testare piu' LLM sulla stessa scena)"
        ),
        default=0,
        min=0,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "seed")
        layout.label(
            text="Flusso manuale: esporta il prompt, usalo nel tuo LLM, incolla qui la risposta.",
            icon="INFO",
        )


def get_prefs(context):
    """Restituisce le preferenze dell'add-on, o None se non disponibili."""
    addons = getattr(getattr(context, "preferences", None), "addons", None)
    if not addons:
        return None
    entry = addons.get(__package__)
    return entry.preferences if entry else None


# ---------------------------------------------------------------------------
# PropertyGroup per la lista override
# ---------------------------------------------------------------------------

class NL2_ObjectOverride(PropertyGroup):
    """Una riga della lista override: contiene nome, stato fisso/mobile, padre ed etichetta."""

    name: StringProperty(name="Object")  # type: ignore

    fixed: BoolProperty(  # type: ignore
        name="Fisso",
        description="Se attivo, l'oggetto non verra' mai spostato dall'add-on",
        default=False,
    )

    parent: StringProperty(  # type: ignore
        name="Padre",
        description=(
            "Nome dell'oggetto padre: questo oggetto si muovera' insieme a lui. "
            "Lascia vuoto per nessun padre"
        ),
        default="",
    )

    label: BoolProperty(  # type: ignore
        name="Etichetta",
        description=(
            "Se attivo, il nome di questo oggetto compare nei render etichettati. "
            "Spegnilo per escluderlo dalle etichette"
        ),
        default=True,
    )

    keep_scale: BoolProperty(  # type: ignore
        name="Gia' in scala",
        description="Se attivo, l'operatore 'Scala a misura reale' NON modifica questo oggetto",
        default=False,
    )


# ---------------------------------------------------------------------------
# UIList per la lista override
# ---------------------------------------------------------------------------

class NL2SCENE3D_UL_overrides(UIList):
    """Lista oggetti: mostra etichetta nei render, padre e lucchetto fisso/mobile."""

    def draw_item(
        self, context, layout, data, item, icon,
        active_data, active_propname, index
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)

            # Toggle ETICHETTA (icona occhio): acceso = etichettato nei render.
            row.prop(
                item, "label",
                text="", toggle=True,
                icon="HIDE_OFF" if item.label else "HIDE_ON",
            )

            row.label(text=item.name, icon="OBJECT_DATA")

            # Ricerca del padre tra gli oggetti della scena (vuoto = nessun padre).
            row.prop_search(
                item, "parent",
                context.scene, "objects",
                text="", icon="CON_CHILDOF",
            )

            # Toggle FISSO/MOBILE (lucchetto).
            row.prop(
                item, "fixed",
                text="", toggle=True,
                icon="LOCKED" if item.fixed else "UNLOCKED",
            )

            # Toggle KEEP_SCALE (puntina).
            row.prop(
                item, "keep_scale",
                text="", toggle=True,
                icon="PINNED" if item.keep_scale else "UNPINNED",
            )

        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.name)


# ---------------------------------------------------------------------------
# Pannello principale nella sidebar
# ---------------------------------------------------------------------------

class NL2SCENE3D_PT_main_panel(Panel):
    """Pannello principale di NL2Scene3D, nella sidebar della 3D View."""

    bl_label      = "NL2Scene3D"
    bl_idname     = "NL2SCENE3D_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category   = "NL2Scene3D"

    def draw(self, context):
        layout = self.layout
        scene  = context.scene
        prefs  = get_prefs(context)

        # Intestazione / verifica add-on abilitato correttamente.
        box = layout.box()
        if prefs is None:
            box.label(text="Add-on non abilitato correttamente", icon="ERROR")
            return
        box.label(text="Flusso LLM manuale (esporta / applica)", icon="SHADERFX")
        
        # Usa il flag a livello modulo invece di fare l'import nel draw().
        if not _PILLOW_AVAILABLE:
            pbox = layout.box()
            pbox.alert = True
            pbox.label(text="Pillow non installato!", icon="ERROR")
            pbox.label(text="I render etichettati non funzioneranno.")
            pbox.operator("nl2scene3d.install_pillow", text="Installa Pillow", icon="IMPORT")

        layout.separator()
        layout.operator(
            "nl2scene3d.inspect",
            text="Inspect Scene (dry-run)",
            icon="VIEWZOOM",
        )
        layout.operator(
            "nl2scene3d.scale_to_real",
            text="Scala a misura reale",
            icon="FULLSCREEN_ENTER",
        )

        # --- Sezione: Override manuali (etichetta / fisso / padre) ---
        layout.separator()
        col = layout.column(align=True)
        col.prop(
            scene, "nl2_overrides_enabled",
            text="Override manuali (etichetta / fisso / padre)",
        )

        if scene.nl2_overrides_enabled:
            # Riga pulsanti: sincronizza, auto-classificazione, svuota.
            row = col.row(align=True)
            row.operator("nl2scene3d.overrides_sync",       text="Sincronizza",  icon="FILE_REFRESH")
            row.operator("nl2scene3d.overrides_autodetect", text="Auto fisso",   icon="ZOOM_SELECTED")
            row.operator("nl2scene3d.overrides_clear",      text="",             icon="TRASH")

            # Riga pulsanti: suggerisci gruppi, accendi/spegni tutte le etichette.
            row2 = col.row(align=True)
            row2.operator(
                "nl2scene3d.overrides_suggest_groups",
                text="Suggerisci gruppi",
                icon="CON_CHILDOF",
            )
            op_on       = row2.operator("nl2scene3d.overrides_labels_all", text="", icon="HIDE_OFF")
            op_on.value = True
            op_off       = row2.operator("nl2scene3d.overrides_labels_all", text="", icon="HIDE_ON")
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
                col.label(
                    text="Occhio = etichetta render, Catena = padre, Lucchetto = fisso/mobile.",
                    icon="INFO",
                )

        # --- Sezione Step 1: disordina e reset ---
        layout.separator()
        col1 = layout.column(align=True)
        col1.label(text="Step 1: Disordina (opzionale)")
        col1.operator("nl2scene3d.randomize",   text="Randomize Layout",  icon="RECOVER_LAST")
        col1.operator("nl2scene3d.reset_home",  text="Reset to Original", icon="LOOP_BACK")

        if scene.nl2_has_home:
            col1.label(text="Originale salvato.", icon="CHECKMARK")
        else:
            col1.label(text="Nessun originale (premi Randomize una volta).", icon="INFO")

        # --- Sezione Step 2: riordina con AI (flusso manuale) ---
        # Numerazione chiarita con suffissi a/b/c per evitare confusione tra lo Step 2 principale e i suoi sotto-step numerati 0/1/2.
        layout.separator()
        col2 = layout.column(align=True)
        col2.label(text="Step 2: Riordina con AI")
        col2.operator(
            "nl2scene3d.render_labeled",
            text="2a. Render con etichette (opzionale)",
            icon="RENDER_STILL",
        )

        # Avvisa se si esporta senza aver prima randomizzato.
        if not scene.nl2_has_home:
            row_warn = col2.row()
            row_warn.alert = True
            row_warn.label(text="Suggerito: esegui prima Randomize (Step 1)", icon="ERROR")

        col2.operator(
            "nl2scene3d.export_for_llm",
            text="2b. Esporta prompt (copia negli appunti)",
            icon="COPYDOWN",
        )

        col2.separator()
        col2.label(text="Incolla il prompt nell'LLM, copia il JSON di risposta, poi:")
        col2.operator(
            "nl2scene3d.apply_from_clipboard",
            text="2c. Applica risposta (dagli appunti)",
            icon="PASTEDOWN",
        )

        col2.separator()
        col2.label(text="Metodi alternativi di applicazione:")
        row3 = col2.row(align=True)
        row3.operator(
            "nl2scene3d.apply_from_text",
            text="Da Text Editor",
            icon="TEXT",
        )
        row3.operator(
            "nl2scene3d.apply_from_file",
            text="Da file",
            icon="FILEBROWSER",
        )


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
    """Registra le classi UI e aggiunge le proprieta' di scena necessarie."""
    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.nl2_overrides = CollectionProperty(type=NL2_ObjectOverride)

    bpy.types.Scene.nl2_overrides_index = IntProperty(default=0)

    bpy.types.Scene.nl2_overrides_enabled = BoolProperty(
        name="Override manuali",
        description=(
            "Se attivo, usa le scelte manuali (etichetta / fisso / padre) "
            "al posto della classificazione automatica"
        ),
        default=False,
    )

    bpy.types.Scene.nl2_has_home = BoolProperty(
        name="Ha stato originale",
        description="Indica se e' stato salvato uno stato originale per il reset",
        default=False,
    )


def unregister():
    """Rimuove le proprieta' di scena e deregistra le classi UI."""
    for attr in (
        "nl2_overrides",
        "nl2_overrides_index",
        "nl2_overrides_enabled",
        "nl2_has_home",
    ):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)

    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass