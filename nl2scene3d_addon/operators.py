# nl2scene3d/operators.py
"""
Operatori Blender. Sono volutamente SOTTILI: orchestrano i moduli del core e
gestiscono UI/errori, senza contenere logica geometrica.
"""

import os
import traceback

import bpy  # type: ignore
from bpy.types import Operator  # type: ignore

from .core import render, reorganizer, scene_io
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


def _label_name_set(context):
    """
    Insieme dei nomi da etichettare nei render. Generico per qualsiasi scena:
      - parte da tutti gli oggetti reali (esclude camere e luci);
      - se nella lista override un oggetto ha 'label' spento, viene escluso.
    Cosi' l'utente accende/spegne l'etichetta per ogni oggetto dal pannello,
    senza dover attivare gli override fisso/mobile.
    """
    scene = context.scene
    names = {o.name for o in scene.objects if o.type not in {"CAMERA", "LIGHT"}}
    items = getattr(scene, "nl2_overrides", None)
    if items and len(items):
        excluded = {e.name for e in items if not e.label}
        names -= excluded
    return names


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
        entry.label = True  # di default etichetta tutto
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


class NL2SCENE3D_OT_overrides_labels_all(Operator):
    """Accende o spegne l'etichetta su TUTTE le voci in un colpo solo."""

    bl_idname  = "nl2scene3d.overrides_labels_all"
    bl_label   = "Etichette: tutte/nessuna"
    bl_options = {"REGISTER", "UNDO"}

    value: bpy.props.BoolProperty(name="Etichetta", default=True)  # type: ignore

    def execute(self, context):
        _sync_overrides(context.scene)
        for entry in context.scene.nl2_overrides:
            entry.label = self.value
        self.report({"INFO"}, f"Etichette {'attivate' if self.value else 'disattivate'} su tutte le voci.")
        return {"FINISHED"}


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

class NL2SCENE3D_OT_scale_to_real(Operator):
    """Scala la scena a misura reale partendo da UN riferimento noto (uniforme)."""

    bl_idname  = "nl2scene3d.scale_to_real"
    bl_label   = "Scala a misura reale"
    bl_options = {"REGISTER", "UNDO"}

    mode: bpy.props.EnumProperty(  # type: ignore
        name="Riferimento",
        items=[
            ("REFERENCE", "Oggetto noto", "Indichi un oggetto e la sua misura reale; il resto segue"),
            ("ROOM", "Dimensione stanza", "Indichi quanto deve essere lungo il lato maggiore della stanza"),
        ],
        default="REFERENCE",
    )
    reference_object: bpy.props.StringProperty(  # type: ignore
        name="Oggetto", description="Oggetto di cui conosci la misura reale",
    )
    target_size: bpy.props.FloatProperty(  # type: ignore
        name="Misura reale (m)",
        description="Lato piu' lungo reale: dell'oggetto scelto (modo Oggetto) o della stanza (modo Stanza)",
        default=2.0, min=0.001, soft_max=20.0,
    )
    scale_structural: bpy.props.BoolProperty(  # type: ignore
        name="Scala anche la stanza/strutturali",
        description="Se spento, lascia stanza/muri e camera/luci come sono e scala solo arredi e oggetti",
        default=False,
    )
    apply_scale: bpy.props.BoolProperty(  # type: ignore
        name="Applica scala (consigliato)", default=True,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=340)

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "mode")
        if self.mode == "REFERENCE":
            col.prop_search(self, "reference_object", context.scene, "objects")
        col.prop(self, "target_size")
        col.prop(self, "scale_structural")
        col.prop(self, "apply_scale")

    def execute(self, context):
        import mathutils  # noqa: PLC0415
        scene = context.scene

        meshes = [o for o in scene.objects if o.type == "MESH"]
        if not meshes:
            self.report({"WARNING"}, "Nessun mesh nella scena.")
            return {"CANCELLED"}

        # AABB unione (mondo) dei mesh: centro-pivot + dimensione per il modo Stanza.
        big = 1.0e9
        mins = mathutils.Vector((big, big, big))
        maxs = mathutils.Vector((-big, -big, -big))
        for o in meshes:
            for c in o.bound_box:
                w = o.matrix_world @ mathutils.Vector(c)
                mins = mathutils.Vector((min(mins.x, w.x), min(mins.y, w.y), min(mins.z, w.z)))
                maxs = mathutils.Vector((max(maxs.x, w.x), max(maxs.y, w.y), max(maxs.z, w.z)))
        center = (mins + maxs) / 2.0

        if self.mode == "REFERENCE":
            ref = scene.objects.get(self.reference_object)
            if ref is None:
                self.report({"ERROR"}, "Scegli un oggetto di riferimento valido.")
                return {"CANCELLED"}
            current = max(ref.dimensions.x, ref.dimensions.y, ref.dimensions.z)
        else:
            current = max(maxs.x - mins.x, maxs.y - mins.y)

        if current <= 1e-9:
            self.report({"ERROR"}, "Dimensione di riferimento nulla: impossibile calcolare il fattore.")
            return {"CANCELLED"}

        factor = float(self.target_size) / float(current)
        if not (factor > 0.0) or factor != factor:  # >0 e non-NaN
            self.report({"ERROR"}, "Fattore di scala non valido.")
            return {"CANCELLED"}

        def is_structural(o):
            dims = [o.dimensions.x, o.dimensions.y, o.dimensions.z]
            cat, _ = default_classification(o.name, o.type, dims)
            return cat == "structural"
        
        targets = []
        keep = {e.name for e in getattr(scene, "nl2_overrides", [])
                if getattr(e, "keep_scale", False)}
        for o in scene.objects:
            if o.name in keep:
                continue
            if not self.scale_structural and (o.type in {"CAMERA", "LIGHT"} or is_structural(o)):
                continue
            targets.append(o)
        if not targets:
            self.report({"WARNING"}, "Nessun oggetto da scalare con queste opzioni.")
            return {"CANCELLED"}

        # Scala uniforme attorno al centro scena: S = T(C) . Scale(f) . T(-C)
        S = (mathutils.Matrix.Translation(center)
             @ mathutils.Matrix.Scale(factor, 4)
             @ mathutils.Matrix.Translation(-center))

        def native_depth(b):
            d, p = 0, b.parent
            while p is not None:
                d += 1
                p = p.parent
            return d

        orig = {o.name: o.matrix_world.copy() for o in targets}
        for o in sorted(targets, key=native_depth):  # padri nativi prima dei figli
            o.matrix_world = S @ orig[o.name]
        try:
            context.view_layer.update()
        except Exception:
            pass

        if self.apply_scale:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
            try:
                bpy.ops.object.select_all(action="DESELECT")
                roots = [o for o in targets if o.type == "MESH" and o.parent is None]
                for o in roots:
                    o.select_set(True)
                if roots:
                    context.view_layer.objects.active = roots[0]
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            except Exception:
                pass

        # Lo stato "originale" salvato non e' piu' coerente con la nuova scala: azzeralo.
        for o in scene.objects:
            for key in ("nl2_home_loc", "nl2_home_rot"):
                if key in o:
                    del o[key]
        scene.nl2_has_home = False

        self.report(
            {"INFO"},
            f"Scena scalata di {factor:.3f}x ({len(targets)} oggetti). "
            f"Rifai Randomize/Export. Stato originale azzerato.",
        )
        return {"FINISHED"}
    
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
                scene_io.capture_pose_snapshot("m_orig")   # originale, per le metriche
                context.scene.nl2_has_home = True
            wm.progress_update(20)
            state = scene_io.extract_scene_state(overrides=_build_overrides(context))

            wm.progress_update(50)
            randomized = SceneRandomizer(seed=seed).randomize(state)

            wm.progress_update(80)
            scene_io.apply_state(randomized)
            scene_io.capture_pose_snapshot("m_rand")        # randomizzato, per le metriche
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


class NL2SCENE3D_OT_render_labeled(Operator):
    """Renderizza le viste (prospettica + pianta) con i nomi degli oggetti sovrapposti."""

    bl_idname  = "nl2scene3d.render_labeled"
    bl_label   = "Render con etichette"
    bl_options = {"REGISTER"}

    add_top_down: bpy.props.BoolProperty(  # type: ignore
        name="Vista dall'alto (pianta)",
        description="Genera anche il render ortografico dall'alto",
        default=True,
    )
    brighten: bpy.props.FloatProperty(  # type: ignore
        name="Luminosita'",
        description="Schiarisce i render (1.0 = invariato, >1 piu' chiaro)",
        default=1.5, min=0.5, max=3.0,
    )
    perspective_lens: bpy.props.FloatProperty(  # type: ignore
        name="Lente prospettica (mm)",
        description="0 = camera invariata. Valori bassi (18-24) allargano il campo (meno zoomata)",
        default=0.0, min=0.0, max=120.0,
    )
    auto_perspective: bpy.props.BoolProperty(  # type: ignore
        name="Auto-inquadratura (tutta la stanza)",
        description="Prospettica da un angolo alto che inquadra l'intera stanza, "
                    "invece della camera della scena",
        default=True,
    )
    auto_lens: bpy.props.FloatProperty(  # type: ignore
        name="Lente auto (mm)",
        description="Lente della camera auto: piu' bassa = campo piu' largo",
        default=24.0, min=10.0, max=50.0,
    )
    add_iso: bpy.props.BoolProperty(  # type: ignore
        name="Vista isometrica",
        description="Aggiunge una vista isometrica ortografica (proporzioni coerenti, senza prospettiva)",
        default=False,
    )

    def execute(self, context):
        try:
            names = _label_name_set(context)
            paths = render.render_labeled_views(
                add_top_down=self.add_top_down,
                add_iso=self.add_iso,
                label_names=names,
                brighten=self.brighten,
                lens_override=(self.perspective_lens if self.perspective_lens > 0 else None),
                auto_perspective=self.auto_perspective,
                auto_lens=self.auto_lens,
            )
            if not paths:
                self.report({"WARNING"}, "Nessun render generato (manca la camera?).")
                return {"CANCELLED"}
            self.report(
                {"INFO"},
                f"Render salvati: {len(paths)} file in {os.path.dirname(paths[0])}",
            )
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Render fallito: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}


class NL2SCENE3D_OT_export_for_llm(Operator):
    """Genera il prompt + JSON della scena e lo scrive in un Text, pronto da copiare nell'LLM."""

    bl_idname  = "nl2scene3d.export_for_llm"
    bl_label   = "Esporta prompt per LLM"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
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
    incollata o file). Rilegge la scena viva, sanifica il JSON dell'LLM e applica."""
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
        # --- metriche di spostamento (O -> R -> C) ---
        try:
            report = scene_io.build_metrics_report()
            mtxt = bpy.data.texts.get("NL2_Metrics") or bpy.data.texts.new("NL2_Metrics")
            mtxt.clear()
            mtxt.write(report)
            print("\n" + report + "\n")
        except Exception:
            traceback.print_exc()

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
    NL2SCENE3D_OT_overrides_labels_all,
    NL2SCENE3D_OT_overrides_clear,
    NL2SCENE3D_OT_reset_home,
    NL2SCENE3D_OT_inspect,
    NL2SCENE3D_OT_scale_to_real,
    NL2SCENE3D_OT_randomize,
    NL2SCENE3D_OT_render_labeled,
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