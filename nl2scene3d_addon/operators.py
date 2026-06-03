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

from .core import reorganizer, scene_io
from .core.classify import default_classification, suggest_grouping
from .core.randomizer import SceneRandomizer


def _get_prefs(context):
    try:
        return context.preferences.addons[__package__].preferences
    except (KeyError, AttributeError):
        return None


def _write_text(name: str, content: str) -> None:
    """Scrive (sovrascrivendo) un Text datablock apribile nel Text Editor."""
    txt = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    txt.clear()
    txt.write(content)


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


_RESPONSE_TEXT = "NL2_AI_Response"  # Text datablock dove l'utente incolla il JSON dell'LLM


class NL2SCENE3D_OT_export_for_llm(Operator):
    """Genera il prompt + JSON della scena e lo scrive in un Text, pronto da copiare nell'LLM."""

    bl_idname  = "nl2scene3d.export_for_llm"
    bl_label   = "Esporta prompt per LLM"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            # Fotografa l'originale al primo intervento (per 'Reset to Original').
            if not context.scene.nl2_has_home:
                scene_io.capture_home_state()
                context.scene.nl2_has_home = True

            state = scene_io.extract_scene_state(overrides=_build_overrides(context))
            roots = [o for o in state.objects if o.is_movable and o.is_root]
            if not roots:
                self.report({"WARNING"}, "Nessun oggetto mobile da riorganizzare.")
                return {"CANCELLED"}

            prompt = reorganizer.build_prompt(state)  # istruzione di default fissa
            _write_text("NL2_AI_Prompt", prompt)

            # Prepara (vuota) la casella in cui incollerai la risposta dell'LLM.
            txt = bpy.data.texts.get(_RESPONSE_TEXT) or bpy.data.texts.new(_RESPONSE_TEXT)
            txt.clear()

            self.report(
                {"INFO"},
                f"Prompt pronto nel Text 'NL2_AI_Prompt' ({len(roots)} oggetti). "
                f"Copialo nell'LLM, poi incolla la risposta nel Text '{_RESPONSE_TEXT}'.",
            )
            return {"FINISHED"}

        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Esportazione fallita: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}


def _apply_llm_response(operator, context, raw_text: str):
    """
    Cuore condiviso della metà 'a valle': identico a cio' che faceva l'AI Reorder
    dopo aver ricevuto la risposta del modello, ma il testo arriva da te (casella
    incollata o file). Rilegge la scena viva, sanifica il JSON e applica.
    """
    wm = context.window_manager
    try:
        wm.progress_begin(0, 100)
        for w in wm.windows:
            w.cursor_set("WAIT")

        # Fotografa l'originale se non e' stato gia' fatto (export o randomize).
        if not context.scene.nl2_has_home:
            scene_io.capture_home_state()
            context.scene.nl2_has_home = True

        wm.progress_update(20)
        state = scene_io.extract_scene_state(overrides=_build_overrides(context))

        wm.progress_update(50)
        parsed = reorganizer.extract_json(raw_text) or {}
        placements = parsed.get("placements")
        n_prop = len(placements) if isinstance(placements, list) else 0
        if n_prop == 0:
            _reset_wm(context)
            operator.report(
                {"WARNING"},
                "Nessuna posizione valida trovata nel JSON. Controlla di aver incollato "
                "la risposta completa del modello (deve contenere \"placements\").",
            )
            return {"CANCELLED"}

        wm.progress_update(80)
        new_state = reorganizer.sanitize_response(state, raw_text)
        counters = scene_io.apply_state(new_state)

        wm.progress_update(100)
        _reset_wm(context)
        operator.report(
            {"INFO"},
            f"Applicato: {n_prop} proposte, {counters['updated']} oggetti spostati. "
            f"'Reset to Original' per annullare.",
        )
        return {"FINISHED"}

    except Exception as exc:  # noqa: BLE001
        _reset_wm(context)
        operator.report({"ERROR"}, f"Applicazione fallita: {exc}")
        traceback.print_exc()
        return {"CANCELLED"}


def _reset_wm(context):
    try:
        wm = context.window_manager
        wm.progress_end()
        for w in wm.windows:
            w.cursor_set("DEFAULT")
    except Exception:
        pass


class NL2SCENE3D_OT_apply_from_text(Operator):
    """Applica la risposta dell'LLM incollata nel Text 'NL2_AI_Response'."""

    bl_idname  = "nl2scene3d.apply_from_text"
    bl_label   = "Applica risposta (dal testo incollato)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        txt = bpy.data.texts.get(_RESPONSE_TEXT)
        if txt is None:
            self.report(
                {"ERROR"},
                f"Text '{_RESPONSE_TEXT}' non trovato. Premi prima 'Esporta prompt per LLM', "
                f"poi incolla lì la risposta del modello.",
            )
            return {"CANCELLED"}
        raw = txt.as_string()
        if not raw.strip():
            self.report({"WARNING"}, f"Il Text '{_RESPONSE_TEXT}' e' vuoto: incolla il JSON del modello.")
            return {"CANCELLED"}
        return _apply_llm_response(self, context, raw)


class NL2SCENE3D_OT_apply_from_file(Operator):
    """Applica la risposta dell'LLM caricandola da un file (.json o .txt)."""

    bl_idname  = "nl2scene3d.apply_from_file"
    bl_label   = "Applica risposta (da file)"
    bl_options = {"REGISTER", "UNDO"}

    # Fornito da invoke(): percorso scelto nel file browser.
    filepath:   bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore
    filter_glob: bpy.props.StringProperty(default="*.json;*.txt", options={"HIDDEN"})  # type: ignore

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not self.filepath:
            self.report({"WARNING"}, "Nessun file selezionato.")
            return {"CANCELLED"}
        try:
            with open(self.filepath, encoding="utf-8") as f:
                raw = f.read()
        except OSError as exc:
            self.report({"ERROR"}, f"Impossibile leggere il file: {exc}")
            return {"CANCELLED"}
        return _apply_llm_response(self, context, raw)


_classes = (
    NL2SCENE3D_OT_overrides_sync,
    NL2SCENE3D_OT_overrides_autodetect,
    NL2SCENE3D_OT_overrides_suggest_groups,
    NL2SCENE3D_OT_overrides_clear,
    NL2SCENE3D_OT_reset_home,
    NL2SCENE3D_OT_inspect,
    NL2SCENE3D_OT_randomize,
    NL2SCENE3D_OT_export_for_llm,
    NL2SCENE3D_OT_apply_from_text,
    NL2SCENE3D_OT_apply_from_file,
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
