# nl2scene3d/operators.py
"""
Operatori Blender. Sono volutamente SOTTILI: orchestrano i moduli del core e
gestiscono UI/errori, senza contenere logica geometrica.

Step 2: solo Randomize (estrai -> randomizza -> applica).
Gli operatori di rendering/AI Reorder arrivano negli step successivi.
"""

import traceback

import bpy  # type: ignore
from bpy.types import Operator  # type: ignore

from .core import scene_io
from .core.classify import default_classification, suggest_grouping
from .core.randomizer import SceneRandomizer


def _get_prefs(context):
    try:
        return context.preferences.addons[__package__].preferences
    except (KeyError, AttributeError):
        return None


def _build_overrides(context):
    """
    Costruisce il dict di override dalle voci della Scene, oppure None se gli
    override manuali sono disattivati o la lista e' vuota. Il core riceve solo
    dati puri (niente bpy): per ogni oggetto lo stato fisso e il padre scelto.
    """
    scene = context.scene
    if not getattr(scene, "nl2_overrides_enabled", False):
        return None
    items = getattr(scene, "nl2_overrides", None)
    if not items or len(items) == 0:
        return None
    return {
        e.name: {"fixed": bool(e.fixed), "parent": e.parent or ""}
        for e in items
    }


def _listable_objects(scene):
    """Oggetti che ha senso elencare negli override (esclude camere e luci)."""
    return [o for o in scene.objects if o.type not in {"CAMERA", "LIGHT"}]


def _sync_overrides(scene) -> tuple[int, int]:
    """
    Allinea la lista override agli oggetti della scena: aggiunge i nuovi (con
    default automatici fisso/mobile), rimuove i mancanti, preserva gli editati.
    Ritorna (aggiunti, rimossi).
    """
    items = scene.nl2_overrides
    existing = {e.name for e in items}
    present = {o.name for o in _listable_objects(scene)}

    added = 0
    for obj in _listable_objects(scene):
        if obj.name in existing:
            continue
        dims = [obj.dimensions.x, obj.dimensions.y, obj.dimensions.z]
        _cat, mov = default_classification(obj.name, obj.type, dims)
        entry = items.add()
        entry.name = obj.name
        entry.fixed = not mov
        added += 1

    removed = 0
    for i in range(len(items) - 1, -1, -1):
        if items[i].name not in present:
            items.remove(i)
            removed += 1

    return added, removed


class NL2SCENE3D_OT_overrides_sync(Operator):
    """Sincronizza la lista con gli oggetti della scena (aggiunge i nuovi, toglie i mancanti)."""

    bl_idname  = "nl2scene3d.overrides_sync"
    bl_label   = "Sincronizza lista override"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        added, removed = _sync_overrides(context.scene)
        self.report({"INFO"}, f"Lista sincronizzata: +{added}, -{removed}.")
        return {"FINISHED"}


class NL2SCENE3D_OT_overrides_autodetect(Operator):
    """Riempie TUTTE le voci con la classificazione automatica (sovrascrive le scelte)."""

    bl_idname  = "nl2scene3d.overrides_autodetect"
    bl_label   = "Rileva automaticamente"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        _sync_overrides(scene)  # assicura che la lista sia completa
        by_name = {o.name: o for o in scene.objects}
        n = 0
        for entry in scene.nl2_overrides:
            obj = by_name.get(entry.name)
            if obj is None:
                continue
            dims = [obj.dimensions.x, obj.dimensions.y, obj.dimensions.z]
            _cat, mov = default_classification(obj.name, obj.type, dims)
            entry.fixed = not mov
            n += 1
        self.report({"INFO"}, f"Auto-classificate {n} voci.")
        return {"FINISHED"}


class NL2SCENE3D_OT_overrides_suggest_groups(Operator):
    """Propone i rapporti padre-figlio in base alla geometria (poi li correggi)."""

    bl_idname  = "nl2scene3d.overrides_suggest_groups"
    bl_label   = "Suggerisci gruppi"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _sync_overrides(context.scene)
            # Estrae lo stato (con gli override fisso/mobile correnti) e propone i gruppi.
            state = scene_io.extract_scene_state(overrides=_build_overrides(context))
            mapping = suggest_grouping(state.objects)

            entries = {e.name: e for e in context.scene.nl2_overrides}
            applied = 0
            for child, parent in mapping.items():
                if child in entries:
                    entries[child].parent = parent
                    applied += 1
            self.report({"INFO"}, f"Proposti {applied} rapporti padre-figlio. Controllali nella lista.")
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Suggerimento gruppi fallito: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}


class NL2SCENE3D_OT_overrides_clear(Operator):
    """Svuota la lista degli override."""

    bl_idname  = "nl2scene3d.overrides_clear"
    bl_label   = "Svuota lista override"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.nl2_overrides.clear()
        self.report({"INFO"}, "Lista override svuotata.")
        return {"FINISHED"}


class NL2SCENE3D_OT_reset_home(Operator):
    """Riporta tutti gli oggetti alla posa originale salvata."""

    bl_idname  = "nl2scene3d.reset_home"
    bl_label   = "Reset to Original"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not scene_io.has_home_state():
            self.report({"WARNING"}, "Nessuno stato originale salvato. Usa Randomize o 'Set as Original'.")
            return {"CANCELLED"}
        try:
            n = scene_io.reset_home_state()
            context.scene.nl2_has_home = True
            self.report({"INFO"}, f"Scena riportata all'originale ({n} oggetti).")
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Reset fallito: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}


class NL2SCENE3D_OT_inspect(Operator):
    """Dry-run: estrae la scena e mostra come viene classificata, SENZA muovere nulla."""

    bl_idname  = "nl2scene3d.inspect"
    bl_label   = "Inspect Scene (dry-run)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            state  = scene_io.extract_scene_state(overrides=_build_overrides(context))
            report = scene_io.format_inspection(state)

            # Scrive il report in un Text datablock apribile nel Text Editor.
            name = "NL2_Inspect_Report"
            txt  = bpy.data.texts.get(name) or bpy.data.texts.new(name)
            txt.clear()
            txt.write(report)

            # E anche in console di sistema, comodo per copiarlo.
            print("\n" + report + "\n")

            n_mov = len(state.movable_objects)
            n_fix = len(state.objects) - n_mov
            self.report(
                {"INFO"},
                f"Inspect: {n_mov} mobili, {n_fix} fissi. "
                f"Apri il Text '{name}' (o la console) per la tabella.",
            )
            return {"FINISHED"}

        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Inspect fallito: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}


class NL2SCENE3D_OT_randomize(Operator):
    """Disordina gli oggetti mobili dentro i confini della stanza."""

    bl_idname  = "nl2scene3d.randomize"
    bl_label   = "Randomize Layout"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        wm = context.window_manager
        try:
            prefs = _get_prefs(context)
            seed  = int(getattr(prefs, "seed", 0)) if prefs else 0

            wm.progress_begin(0, 100)
            for w in wm.windows:
                w.cursor_set("WAIT")

            # Fotografa l'originale al primo Randomize (prima di disordinare),
            # cosi' "Reset to Original" potra' sempre tornare alla scena pristina.
            if not context.scene.nl2_has_home:
                scene_io.capture_home_state()
                context.scene.nl2_has_home = True

            wm.progress_update(20)
            state = scene_io.extract_scene_state(overrides=_build_overrides(context))

            wm.progress_update(50)
            randomized = SceneRandomizer(seed=seed).randomize(state)

            wm.progress_update(80)
            scene_io.apply_state(randomized)

            wm.progress_update(100)
            self._reset_ui(context)

            n = len(state.movable_objects)
            self.report({"INFO"}, f"Disordinati {n} oggetti mobili.")
            return {"FINISHED"}

        except Exception as exc:  # noqa: BLE001
            self._reset_ui(context)
            self.report({"ERROR"}, f"Randomize fallito: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}

    @staticmethod
    def _reset_ui(context):
        try:
            wm = context.window_manager
            wm.progress_end()
            for w in wm.windows:
                w.cursor_set("DEFAULT")
        except Exception:
            pass


_classes = (
    NL2SCENE3D_OT_overrides_sync,
    NL2SCENE3D_OT_overrides_autodetect,
    NL2SCENE3D_OT_overrides_suggest_groups,
    NL2SCENE3D_OT_overrides_clear,
    NL2SCENE3D_OT_reset_home,
    NL2SCENE3D_OT_inspect,
    NL2SCENE3D_OT_randomize,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
