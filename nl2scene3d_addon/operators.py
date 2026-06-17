# nl2scene3d/operators.py
"""
Operatori Blender dell'add-on NL2Scene3D.

Gli operatori sono volutamente sottili: orchestrano i moduli del core e
gestiscono UI ed errori, senza contenere logica geometrica diretta.

Struttura:
    - Funzioni helper private (_get_prefs, _write_text, _build_overrides, ...)
    - Operatori override (sync, autodetect, suggest_groups, labels, clear)
    - Operatori principali (reset_home, inspect, scale_to_real, randomize,
      render_labeled, export_for_llm, apply_from_text, apply_from_file)
    - Funzioni register / unregister
"""

import os
import threading
import traceback

import bpy  # type: ignore
from bpy.types import Operator  # type: ignore

from . import llm_providers
from .core import render, reorganizer, scene_io
from .core.classify import default_classification, suggest_grouping
from .core.randomizer import SceneRandomizer
from .core.models import RoomBounds


# ---------------------------------------------------------------------------
# Helper privati
# ---------------------------------------------------------------------------

def _get_prefs(context):
    """Restituisce le preferenze dell'add-on, o None se non disponibili."""
    try:
        return context.preferences.addons[__package__].preferences
    except (KeyError, AttributeError):
        return None


def _write_text(name: str, content: str) -> None:
    """
    Scrive (sovrascrivendo) un Text datablock nel Text Editor di Blender.
    Se il datablock non esiste viene creato automaticamente.
    """
    txt = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    txt.clear()
    txt.write(content)


def _build_overrides(context):
    """
    Costruisce il dizionario di override a partire dalle voci della Scene,
    oppure restituisce None se gli override manuali sono disattivati o la
    lista e' vuota.

    Il core riceve solo dati puri (niente bpy): per ogni oggetto vengono
    trasmessi lo stato fisso/mobile e l'eventuale padre scelto dall'utente.
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
    Restituisce l'insieme dei nomi degli oggetti da etichettare nei render.

    Logica generica per qualsiasi scena:
        - Parte da tutti gli oggetti reali (esclude camere e luci).
        - Se nella lista override un oggetto ha 'label' spento, viene escluso.

    In questo modo l'utente puo' accendere o spegnere l'etichetta per ogni
    oggetto dal pannello, indipendentemente dalle impostazioni fisso/mobile.
    """
    scene = context.scene
    names = {o.name for o in scene.objects if o.type not in {"CAMERA", "LIGHT"}}

    items = getattr(scene, "nl2_overrides", None)
    if items and len(items):
        excluded = {e.name for e in items if not e.label}
        names -= excluded

    return names


def _listable_objects(scene):
    """Restituisce gli oggetti che ha senso elencare negli override (esclude camere e luci)."""
    return [o for o in scene.objects if o.type not in {"CAMERA", "LIGHT"}]


def _sync_overrides(scene) -> tuple[int, int]:
    """
    Allinea la lista override agli oggetti della scena corrente:
        - Aggiunge le voci mancanti (con classificazione automatica fisso/mobile).
        - Rimuove le voci di oggetti che non esistono piu'.
        - Preserva le voci gia' modificate dall'utente.

    Restituisce una tupla (aggiunti, rimossi).
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
        entry.label = True  # etichettato nei render per default
        added += 1

    removed = 0
    for i in range(len(items) - 1, -1, -1):
        if items[i].name not in present:
            items.remove(i)
            removed += 1

    return added, removed


# ---------------------------------------------------------------------------
# Operatori: gestione override
# ---------------------------------------------------------------------------

class NL2SCENE3D_OT_overrides_sync(Operator):
    """Sincronizza la lista override con gli oggetti della scena (aggiunge i nuovi, toglie i mancanti)."""

    bl_idname = "nl2scene3d.overrides_sync"
    bl_label = "Sincronizza lista override"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        added, removed = _sync_overrides(context.scene)
        self.report({"INFO"}, f"Lista sincronizzata: +{added}, -{removed}.")
        return {"FINISHED"}


class NL2SCENE3D_OT_overrides_autodetect(Operator):
    """Riempie tutte le voci con la classificazione automatica (sovrascrive le scelte manuali)."""

    bl_idname = "nl2scene3d.overrides_autodetect"
    bl_label = "Rileva automaticamente"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        _sync_overrides(scene)  # assicura che la lista sia completa prima di procedere

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
    """Propone i rapporti padre-figlio in base alla geometria (da correggere manualmente)."""

    bl_idname = "nl2scene3d.overrides_suggest_groups"
    bl_label = "Suggerisci gruppi"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _sync_overrides(context.scene)

            # Estrae lo stato corrente (con gli override fisso/mobile attivi)
            # e propone le relazioni padre-figlio basate sulla geometria.
            state = scene_io.extract_scene_state(overrides=_build_overrides(context))
            mapping = suggest_grouping(state.objects)

            entries = {e.name: e for e in context.scene.nl2_overrides}
            applied = 0
            for child, parent in mapping.items():
                if child in entries:
                    entries[child].parent = parent
                    applied += 1

            self.report(
                {"INFO"},
                f"Proposti {applied} rapporti padre-figlio. Controllali nella lista.",
            )
            return {"FINISHED"}

        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Suggerimento gruppi fallito: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}


class NL2SCENE3D_OT_overrides_labels_all(Operator):
    """Accende o spegne l'etichetta su tutte le voci della lista in un colpo solo."""

    bl_idname = "nl2scene3d.overrides_labels_all"
    bl_label = "Etichette: tutte/nessuna"
    bl_options = {"REGISTER", "UNDO"}

    value: bpy.props.BoolProperty(name="Etichetta", default=True)  # type: ignore

    def execute(self, context):
        _sync_overrides(context.scene)
        for entry in context.scene.nl2_overrides:
            entry.label = self.value
        self.report(
            {"INFO"},
            f"Etichette {'attivate' if self.value else 'disattivate'} su tutte le voci.",
        )
        return {"FINISHED"}


class NL2SCENE3D_OT_overrides_clear(Operator):
    """Svuota completamente la lista degli override."""

    bl_idname = "nl2scene3d.overrides_clear"
    bl_label = "Svuota lista override"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.nl2_overrides.clear()
        self.report({"INFO"}, "Lista override svuotata.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operatori: stato originale
# ---------------------------------------------------------------------------

class NL2SCENE3D_OT_reset_home(Operator):
    """Riporta tutti gli oggetti alla posa originale salvata."""

    bl_idname = "nl2scene3d.reset_home"
    bl_label = "Reset to Original"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not scene_io.has_home_state():
            self.report(
                {"WARNING"},
                "Nessuno stato originale salvato. Usa Randomize o 'Set as Original'.",
            )
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


# ---------------------------------------------------------------------------
# Operatori: ispezione
# ---------------------------------------------------------------------------

class NL2SCENE3D_OT_inspect(Operator):
    """Dry-run: estrae la scena e mostra come viene classificata, senza muovere nulla."""

    bl_idname = "nl2scene3d.inspect"
    bl_label = "Inspect Scene (dry-run)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            state = scene_io.extract_scene_state(overrides=_build_overrides(context))
            report = scene_io.format_inspection(state)

            # Scrive il report in un Text datablock consultabile nel Text Editor.
            name = "NL2_Inspect_Report"
            txt = bpy.data.texts.get(name) or bpy.data.texts.new(name)
            txt.clear()
            txt.write(report)

            # Stampa anche in console di sistema per facilitare la copia.
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


# ---------------------------------------------------------------------------
# Operatori: scala
# ---------------------------------------------------------------------------

class NL2SCENE3D_OT_scale_to_real(Operator):
    """Scala la scena a misura reale partendo da un unico riferimento noto (scala uniforme)."""

    bl_idname = "nl2scene3d.scale_to_real"
    bl_label = "Scala a misura reale"
    bl_options = {"REGISTER", "UNDO"}

    mode: bpy.props.EnumProperty(  # type: ignore
        name="Riferimento",
        items=[
            (
                "REFERENCE",
                "Oggetto noto",
                "Indica un oggetto e la sua misura reale; il resto segue proporzionalmente",
            ),
            (
                "ROOM",
                "Dimensione stanza",
                "Indica quanto deve essere lungo il lato maggiore della stanza",
            ),
        ],
        default="REFERENCE",
    )

    reference_object: bpy.props.StringProperty(  # type: ignore
        name="Oggetto",
        description="Oggetto di cui si conosce la misura reale",
    )

    target_size: bpy.props.FloatProperty(  # type: ignore
        name="Misura reale (m)",
        description=(
            "Lato piu' lungo reale: dell'oggetto scelto (modo Oggetto) "
            "o della stanza (modo Stanza)"
        ),
        default=2.0,
        min=0.001,
        soft_max=20.0,
    )

    scale_structural: bpy.props.BoolProperty(  # type: ignore
        name="Scala anche la stanza/strutturali",
        description=(
            "Se spento, lascia stanza/muri e camera/luci invariati "
            "e scala solo arredi e oggetti"
        ),
        default=False,
    )

    apply_scale: bpy.props.BoolProperty(  # type: ignore
        name="Applica scala (consigliato)",
        default=True,
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

        # Calcola l'AABB unione (spazio mondo) di tutti i mesh:
        # serve il centro-pivot per il modo Stanza e la dimensione di riferimento.
        big = 1.0e9
        mins = mathutils.Vector((big, big, big))
        maxs = mathutils.Vector((-big, -big, -big))
        for o in meshes:
            for c in o.bound_box:
                w = o.matrix_world @ mathutils.Vector(c)
                mins = mathutils.Vector((min(mins.x, w.x), min(mins.y, w.y), min(mins.z, w.z)))
                maxs = mathutils.Vector((max(maxs.x, w.x), max(maxs.y, w.y), max(maxs.z, w.z)))

        center = (mins + maxs) / 2.0

        # Determina la dimensione corrente in base alla modalita' scelta.
        if self.mode == "REFERENCE":
            ref = scene.objects.get(self.reference_object)
            if ref is None:
                self.report({"ERROR"}, "Scegli un oggetto di riferimento valido.")
                return {"CANCELLED"}
            current = max(ref.dimensions.x, ref.dimensions.y, ref.dimensions.z)
        else:
            current = max(maxs.x - mins.x, maxs.y - mins.y)

        if current <= 1e-9:
            self.report(
                {"ERROR"},
                "Dimensione di riferimento nulla: impossibile calcolare il fattore.",
            )
            return {"CANCELLED"}

        factor = float(self.target_size) / float(current)

        # Verifica che il fattore sia un numero finito positivo.
        if not (factor > 0.0) or factor != factor:  # >0 e non-NaN
            self.report({"ERROR"}, "Fattore di scala non valido.")
            return {"CANCELLED"}

        def is_structural(o):
            """True se l'oggetto e' classificato come strutturale."""
            dims = [o.dimensions.x, o.dimensions.y, o.dimensions.z]
            cat, _ = default_classification(o.name, o.type, dims)
            return cat == "structural"

        # Costruisce la lista degli oggetti da scalare, escludendo:
        #   - oggetti con 'keep_scale' attivo negli override;
        #   - camera/luci e strutturali (se scale_structural e' spento).
        targets = []
        keep = {
            e.name
            for e in getattr(scene, "nl2_overrides", [])
            if getattr(e, "keep_scale", False)
        }
        for o in scene.objects:
            if o.name in keep:
                continue
            if not self.scale_structural and (
                    o.type in {"CAMERA", "LIGHT"} or is_structural(o)
            ):
                continue
            targets.append(o)

        if not targets:
            self.report({"WARNING"}, "Nessun oggetto da scalare con queste opzioni.")
            return {"CANCELLED"}

        # Scala uniforme attorno al centro della scena:
        #   S = T(center) . Scale(factor) . T(-center)
        S = (
                mathutils.Matrix.Translation(center)
                @ mathutils.Matrix.Scale(factor, 4)
                @ mathutils.Matrix.Translation(-center)
        )

        def native_depth(b):
            """Profondita' del parent nativo (0 = nessun parent)."""
            d, p = 0, b.parent
            while p is not None:
                d += 1
                p = p.parent
            return d

        # Applica la trasformazione: i padri nativi devono essere processati
        # prima dei figli per evitare doppia applicazione.
        orig = {o.name: o.matrix_world.copy() for o in targets}
        for o in sorted(targets, key=native_depth):
            o.matrix_world = S @ orig[o.name]

        try:
            context.view_layer.update()
        except Exception:
            pass

        # Applica la scala agli oggetti mesh radice (consigliato per evitare
        # scale non unitarie che complicano le operazioni future).
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
                bpy.ops.object.transform_apply(
                    location=False, rotation=False, scale=True
                )
            except Exception:
                pass

        # Lo stato "originale" salvato e gli snapshot delle metriche non sono
        # piu' coerenti con la nuova scala: vengono azzerati per evitare reset
        # errati e metriche di spostamento prive di senso (pose in scale diverse).
        stale_keys = (
            "nl2_home_loc", "nl2_home_rot",
            "nl2_m_orig_loc", "nl2_m_orig_rot",
            "nl2_m_rand_loc", "nl2_m_rand_rot",
        )
        for o in scene.objects:
            for key in stale_keys:
                if key in o:
                    del o[key]
        scene.nl2_has_home = False

        self.report(
            {"INFO"},
            f"Scena scalata di {factor:.3f}x ({len(targets)} oggetti). "
            f"Rifai Randomize/Export. Stato originale azzerato.",
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operatori: randomizzazione
# ---------------------------------------------------------------------------

class NL2SCENE3D_OT_randomize(Operator):
    """Disordina gli oggetti mobili all'interno dei confini della stanza."""

    bl_idname = "nl2scene3d.randomize"
    bl_label = "Randomize Layout"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        wm = context.window_manager
        try:
            prefs = _get_prefs(context)
            seed = int(getattr(prefs, "seed", 0)) if prefs else 0

            wm.progress_begin(0, 100)
            for w in wm.windows:
                w.cursor_set("WAIT")

            # Fotografa lo stato originale al primo Randomize (prima di disordinare),
            # cosi' "Reset to Original" potra' sempre tornare alla scena pristina.
            if not context.scene.nl2_has_home:
                scene_io.capture_home_state()
                # Snapshot originale per le metriche: non sovrascrivere se gia'
                # presente (es. .blend riaperto dopo un Randomize precedente).
                scene_io.capture_pose_snapshot("m_orig", overwrite=False)
                context.scene.nl2_has_home = True

            wm.progress_update(20)
            state = scene_io.extract_scene_state(overrides=_build_overrides(context))

            wm.progress_update(50)
            randomized = SceneRandomizer(seed=seed).randomize(state)

            wm.progress_update(80)
            scene_io.apply_state(randomized)
            scene_io.capture_pose_snapshot("m_rand")  # snapshot randomizzato per le metriche

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
        """Ripristina la barra di avanzamento e il cursore a fine operazione."""
        try:
            wm = context.window_manager
            wm.progress_end()
            for w in wm.windows:
                w.cursor_set("DEFAULT")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Operatori: render
# ---------------------------------------------------------------------------

# Nome del Text datablock in cui l'utente incolla la risposta JSON dell'LLM.
_RESPONSE_TEXT = "NL2_AI_Response"


class NL2SCENE3D_OT_render_labeled(Operator):
    """Renderizza le viste della scena (prospettica + pianta) con i nomi degli oggetti sovrapposti."""

    bl_idname = "nl2scene3d.render_labeled"
    bl_label = "Render con etichette"
    bl_options = {"REGISTER"}

    add_top_down: bpy.props.BoolProperty(  # type: ignore
        name="Vista dall'alto (pianta)",
        description="Genera anche il render ortografico dall'alto",
        default=True,
    )

    brighten: bpy.props.FloatProperty(  # type: ignore
        name="Luminosita'",
        description="Schiarisce i render (1.0 = invariato, valori maggiori = piu' chiaro)",
        default=1.5,
        min=0.5,
        max=3.0,
    )

    perspective_lens: bpy.props.FloatProperty(  # type: ignore
        name="Lente prospettica (mm)",
        description=(
            "0 = camera invariata. Valori bassi (18-24) allargano il campo visivo "
            "(meno zoomata)"
        ),
        default=0.0,
        min=0.0,
        max=120.0,
    )

    auto_perspective: bpy.props.BoolProperty(  # type: ignore
        name="Auto-inquadratura (tutta la stanza)",
        description=(
            "Vista prospettica da un angolo alto che inquadra l'intera stanza, "
            "invece di usare la camera della scena"
        ),
        default=True,
    )

    auto_lens: bpy.props.FloatProperty(  # type: ignore
        name="Lente auto (mm)",
        description="Lente della camera auto: valori piu' bassi = campo visivo piu' largo",
        default=24.0,
        min=10.0,
        max=50.0,
    )

    add_iso: bpy.props.BoolProperty(  # type: ignore
        name="Vista isometrica",
        description=(
            "Aggiunge una vista isometrica ortografica "
            "(proporzioni coerenti, senza distorsione prospettica)"
        ),
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
                lens_override=(
                    self.perspective_lens if self.perspective_lens > 0 else None
                ),
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


# ---------------------------------------------------------------------------
# Operatori: esportazione ed applicazione risposta LLM
# ---------------------------------------------------------------------------

class NL2SCENE3D_OT_export_for_llm(Operator):
    """Genera il prompt + JSON della scena e lo scrive in un Text datablock, pronto da copiare nell'LLM."""

    bl_idname = "nl2scene3d.export_for_llm"
    bl_label = "Esporta prompt per LLM"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            # Salva lo stato originale se non e' ancora stato fatto.
            if not context.scene.nl2_has_home:
                scene_io.capture_home_state()
                context.scene.nl2_has_home = True

            state = scene_io.extract_scene_state(overrides=_build_overrides(context))
            roots = [o for o in state.objects if o.is_movable and o.is_root]

            if not roots:
                self.report({"WARNING"}, "Nessun oggetto mobile da riorganizzare.")
                return {"CANCELLED"}

            # Genera il prompt con le istruzioni fisse, i dati della scena e linee guida personalizzate.
            custom_inst = getattr(context.scene, "nl2_custom_instructions", "")
            prompt = reorganizer.build_prompt(state, custom_instructions=custom_inst)
            _write_text("NL2_AI_Prompt", prompt)

            # Prepara (vuoto) il Text in cui l'utente incollerà la risposta.
            txt = bpy.data.texts.get(_RESPONSE_TEXT) or bpy.data.texts.new(_RESPONSE_TEXT)
            txt.clear()

            # Copia automaticamente negli appunti di sistema
            context.window_manager.clipboard = prompt

            self.report(
                {"INFO"},
                f"Prompt copiato negli APPUNTI e salvato in 'NL2_AI_Prompt' ({len(roots)} oggetti). "
                f"Incollalo (Ctrl+V) direttamente nel tuo LLM!",
            )
            return {"FINISHED"}

        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Esportazione fallita: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}


def _apply_llm_response(operator, context, raw_text: str):
    """
    Logica condivisa per applicare la risposta dell'LLM alla scena.

    Riceve il testo grezzo (incollato o caricato da file), rilancia la
    scena corrente, sanifica il JSON proposto dall'LLM e applica le nuove
    posizioni rispettando tutti i vincoli geometrici.

    Usato sia da NL2SCENE3D_OT_apply_from_text sia da NL2SCENE3D_OT_apply_from_file.
    """
    wm = context.window_manager
    try:
        wm.progress_begin(0, 100)
        for w in wm.windows:
            w.cursor_set("WAIT")

        # Salva lo stato originale se non e' ancora stato fatto
        # (caso in cui si applica senza aver prima randomizzato).
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
                'la risposta completa del modello (deve contenere "placements").',
            )
            return {"CANCELLED"}

        wm.progress_update(80)
        new_state = reorganizer.sanitize_response(state, parsed)
        counters = scene_io.apply_state(new_state)

        # Genera e salva il report delle metriche di spostamento (O -> R -> C).
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
    """Ripristina la barra di avanzamento e il cursore del window manager."""
    try:
        wm = context.window_manager
        wm.progress_end()
        for w in wm.windows:
            w.cursor_set("DEFAULT")
    except Exception:
        pass


class NL2SCENE3D_OT_apply_from_text(Operator):
    """Applica la risposta dell'LLM incollata nel Text datablock 'NL2_AI_Response'."""

    bl_idname = "nl2scene3d.apply_from_text"
    bl_label = "Applica risposta (dal testo incollato)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        txt = bpy.data.texts.get(_RESPONSE_TEXT)
        if txt is None:
            self.report(
                {"ERROR"},
                f"Text '{_RESPONSE_TEXT}' non trovato. Premi prima 'Esporta prompt per LLM', "
                f"poi incolla la risposta del modello.",
            )
            return {"CANCELLED"}

        raw = txt.as_string()
        if not raw.strip():
            self.report(
                {"WARNING"},
                f"Il Text '{_RESPONSE_TEXT}' e' vuoto: incolla il JSON del modello.",
            )
            return {"CANCELLED"}

        return _apply_llm_response(self, context, raw)


class NL2SCENE3D_OT_apply_from_clipboard(Operator):
    """Applica la risposta dell'LLM leggendola direttamente dagli appunti di sistema."""

    bl_idname = "nl2scene3d.apply_from_clipboard"
    bl_label = "Applica risposta (dagli appunti)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        raw = context.window_manager.clipboard
        if not raw or not raw.strip():
            self.report(
                {"WARNING"},
                "Gli appunti di sistema sono vuoti. Copia prima la risposta JSON dall'LLM.",
            )
            return {"CANCELLED"}

        return _apply_llm_response(self, context, raw)


class NL2SCENE3D_OT_apply_from_file(Operator):
    """Applica la risposta dell'LLM caricandola da un file (.json o .txt)."""

    bl_idname = "nl2scene3d.apply_from_file"
    bl_label = "Applica risposta (da file)"
    bl_options = {"REGISTER", "UNDO"}

    # Il percorso viene fornito da invoke() tramite il file browser di Blender.
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore
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


class NL2SCENE3D_OT_reorganize_with_api(Operator):
    """Riorganizza la scena in automatico: renderizza, chiama l'LLM via API e applica la risposta."""

    bl_idname = "nl2scene3d.reorganize_with_api"
    bl_label = "Riordina con AI (chiamata API automatica)"
    bl_options = {"REGISTER", "UNDO"}

    # Stato interno della chiamata in background (non sono bpy.props: vivono
    # solo per la durata dell'istanza modale).
    _thread = None
    _timer = None
    _holder = None  # dict condiviso col thread: {"result": LLMResult|None, "error": Exception|None}

    def execute(self, context):
        # execute() viene usato anche se l'operatore e' chiamato da script senza
        # finestra: in quel caso la chiamata e' sincrona (bloccante).
        return self._start(context, blocking=True)

    def invoke(self, context, event):
        # In UI usiamo il flusso modale + thread, cosi' Blender non si congela
        # durante l'attesa della risposta dell'LLM.
        return self._start(context, blocking=False)

    # -- avvio -------------------------------------------------------------
    def _start(self, context, blocking: bool):
        prefs = _get_prefs(context)
        if prefs is None:
            self.report({"ERROR"}, "Preferenze add-on non disponibili.")
            return {"CANCELLED"}

        provider = getattr(prefs, "llm_provider", llm_providers.GEMINI)
        api_key = _active_api_key(prefs, provider)
        model = getattr(prefs, "llm_model", "").strip()
        temp = float(getattr(prefs, "llm_temperature", 0.7))
        timeout = float(getattr(prefs, "llm_timeout", 120.0))

        # Validazione preliminare della chiave (messaggio chiaro prima di partire).
        try:
            llm_providers._resolve_key(provider, api_key)
        except llm_providers.LLMError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        try:
            # Salva lo stato originale se non e' ancora stato fatto.
            if not context.scene.nl2_has_home:
                scene_io.capture_home_state()
                context.scene.nl2_has_home = True

            state = scene_io.extract_scene_state(overrides=_build_overrides(context))
            roots = [o for o in state.objects if o.is_movable and o.is_root]
            if not roots:
                self.report({"WARNING"}, "Nessun oggetto mobile da riorganizzare.")
                return {"CANCELLED"}

            custom_inst = getattr(context.scene, "nl2_custom_instructions", "")
            prompt = reorganizer.build_prompt(state, custom_instructions=custom_inst)
            _write_text("NL2_AI_Prompt", prompt)

            # Render fresco delle viste etichettate (sul main thread: usa bpy).
            image_paths: list[str] = []
            if getattr(prefs, "llm_auto_render", True):
                try:
                    names = _label_name_set(context)
                    image_paths = render.render_labeled_views(
                        add_top_down=True,
                        add_iso=False,
                        label_names=names,
                        auto_perspective=True,
                    ) or []
                except Exception:
                    # Il render non e' indispensabile: l'LLM puo' lavorare sul
                    # solo JSON, anche se con meno contesto visivo.
                    traceback.print_exc()
                    image_paths = []
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Preparazione fallita: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}

        # Parametri della chiamata, congelati per il thread.
        call_kwargs = dict(
            provider=provider, api_key=api_key, model=model,
            prompt=prompt, image_paths=image_paths,
            temperature=temp, timeout=timeout, verbose=True,
            max_retries=int(getattr(prefs, "llm_max_retries", 4)),
        )

        print(
            f"[NL2Scene3D] Riordino automatico: {len(roots)} oggetti mobili, "
            f"{len(image_paths)} render allegati, provider={provider}.",
            flush=True,
        )

        if blocking:
            try:
                result = llm_providers.call_llm(**call_kwargs)
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, f"Chiamata LLM fallita: {exc}")
                return {"CANCELLED"}
            return _apply_llm_response(self, context, result.text)

        # --- modalita' UI: thread + timer ---
        self._holder = {"result": None, "error": None}

        def _worker():
            print("[NL2Scene3D] Thread di rete avviato, chiamo l'LLM...", flush=True)
            try:
                self._holder["result"] = llm_providers.call_llm(**call_kwargs)
                print("[NL2Scene3D] Thread di rete: risposta ottenuta.", flush=True)
            except Exception as exc:  # noqa: BLE001  (catturata e mostrata nel modal)
                self._holder["error"] = exc
                print(f"[NL2Scene3D] Thread di rete: ERRORE -> {exc}", flush=True)
                traceback.print_exc()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

        wm = context.window_manager
        for w in wm.windows:
            w.cursor_set("WAIT")
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)

        n_imgs = len(image_paths)
        self.report(
            {"INFO"},
            f"Chiamata a {provider} ({model or llm_providers.DEFAULT_MODELS[provider]}) "
            f"con {n_imgs} immagini... attendi.",
        )
        return {"RUNNING_MODAL"}

    # -- loop modale -------------------------------------------------------
    def modal(self, context, event):
        if event.type != "TIMER":
            # Lasciamo passare gli altri eventi senza bloccare la UI; la
            # richiesta di rete non e' interrompibile in modo pulito, quindi
            # non gestiamo ESC come annullamento.
            return {"PASS_THROUGH"}

        if self._thread is not None and self._thread.is_alive():
            return {"RUNNING_MODAL"}

        # Thread terminato: pulizia timer/cursore e gestione del risultato.
        self._finish_ui(context)

        error = (self._holder or {}).get("error")
        result = (self._holder or {}).get("result")

        if error is not None:
            self.report({"ERROR"}, f"Chiamata LLM fallita: {error}")
            return {"CANCELLED"}

        if result is None:
            self.report({"ERROR"}, "Nessuna risposta dall'LLM.")
            return {"CANCELLED"}

        # Salva la risposta grezza per ispezione/debug, poi applica con la
        # stessa pipeline di sanitizzazione del flusso manuale.
        try:
            _write_text(_RESPONSE_TEXT, result.text)
        except Exception:
            pass

        print(
            f"[NL2Scene3D] Applico la risposta ({len(result.text)} char) "
            "con la pipeline di sanitizzazione...",
            flush=True,
        )
        return _apply_llm_response(self, context, result.text)

    # -- cleanup -----------------------------------------------------------
    def _finish_ui(self, context):
        wm = context.window_manager
        try:
            if self._timer is not None:
                wm.event_timer_remove(self._timer)
        except Exception:
            pass
        self._timer = None
        try:
            for w in wm.windows:
                w.cursor_set("DEFAULT")
        except Exception:
            pass


def _active_api_key(prefs, provider: str) -> str:
    """Restituisce la chiave API delle preferenze per il provider selezionato."""
    field = {
        llm_providers.GEMINI: "gemini_api_key",
        llm_providers.ANTHROPIC: "anthropic_api_key",
        llm_providers.OPENAI: "openai_api_key",
    }.get(provider, "gemini_api_key")
    return getattr(prefs, field, "") or ""


class NL2SCENE3D_OT_install_pillow(Operator):
    """Installa Pillow nel Python interno di Blender usando pip --user."""

    bl_idname = "nl2scene3d.install_pillow"
    bl_label = "Installa Pillow automaticamente"
    bl_options = {"REGISTER"}

    def execute(self, context):
        import importlib
        import subprocess
        import sys
        import os
        import traceback

        self.report({"INFO"}, "Avvio installazione di Pillow...")

        # Rilevamento robusto dell'eseguibile Python di Blender
        #
        # In Blender, sys.executable punta normalmente al Python embedded
        # (es. .../blender/4.x/python/bin/python3.xx). In alcune build/distro
        # punta invece al binario di Blender. La strategia e':
        #   1. Usa sys.executable come prima scelta.
        #   2. Verifica che sia un interprete Python valido (--version).
        #   3. Se non lo e', cerca nel prefisso Python candidati multipiattaforma.
        #      La lista include percorsi macOS (.app bundle), Linux e Windows.
        python_exe = sys.executable

        def _is_python(path: str) -> bool:
            """True se il binario risponde correttamente a `python --version`."""
            try:
                out = subprocess.check_output(
                    [path, "--version"], stderr=subprocess.STDOUT, timeout=5
                )
                return out.lower().startswith(b"python")
            except Exception:
                return False

        if not _is_python(python_exe):
            # sys.executable non e' un interprete Python (e' il launcher di Blender).
            # Cerca il Python embedded nel prefisso del runtime corrente.
            candidates = [
                os.path.join(sys.prefix, "bin", "python.exe"),  # Windows
                os.path.join(sys.prefix, "python.exe"),  # Windows (alt)
                os.path.join(sys.prefix, "bin", "python3"),  # Linux/macOS
                os.path.join(sys.prefix, "bin", "python"),  # Linux/macOS (alt)
                # macOS .app bundle: il Python e' annidato piu' in profondita'.
                os.path.join(sys.prefix, "..", "bin", "python3"),
                os.path.join(sys.prefix, "..", "..", "bin", "python3"),
            ]
            found = False
            for path in candidates:
                path = os.path.normpath(path)
                if os.path.isfile(path) and _is_python(path):
                    python_exe = path
                    found = True
                    break
            if not found:
                self.report(
                    {"ERROR"},
                    "Impossibile trovare l'interprete Python di Blender. "
                    "Installa Pillow manualmente con: "
                    f"`{sys.executable} -m pip install Pillow --user`",
                )
                return {"CANCELLED"}

        try:
            # Installa Pillow in spazio utente.
            subprocess.check_call([python_exe, "-m", "pip", "install", "Pillow", "--user"])

            # Assicura che il site-packages utente sia in sys.path per il reload.
            import site
            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                sys.path.append(user_site)
            importlib.invalidate_caches()
            from PIL import Image  # noqa: F401

            self.report({"INFO"}, "Pillow installato correttamente! Ora puoi generare i render etichettati.")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Installazione fallita: {exc}")
            traceback.print_exc()
            return {"CANCELLED"}


class NL2SCENE3D_OT_create_bounds_helper(Operator):
    """Crea o seleziona un cubo di riferimento wireframe per definire manualmente i confini della stanza."""

    bl_idname = "nl2scene3d.create_bounds_helper"
    bl_label = "Crea Helper Confini"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        import mathutils
        scene = context.scene
        helper_name = "NL2_RoomBounds_Helper"

        # Se esiste gia', lo selezioniamo
        helper = bpy.data.objects.get(helper_name)
        if helper:
            # Assicura che sia visibile e selezionato
            helper.hide_viewport = False
            helper.hide_select = False
            for o in context.selected_objects:
                o.select_set(False)
            helper.select_set(True)
            context.view_layer.objects.active = helper
            self.report({"INFO"}, "Helper confini esistente selezionato.")
            return {"FINISHED"}

        # Altrimenti lo creiamo basandoci sui confini correnti
        # Estrarre lo stato attuale senza overrides manuali per avere una stima
        try:
            state = scene_io.extract_scene_state()
            bounds = state.room_bounds
        except Exception:
            bounds = RoomBounds(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, z_floor=0.0, z_ceiling=3.0)

        # Crea mesh cubo
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        helper = context.active_object
        helper.name = helper_name
        helper.display_type = "WIRE"
        helper.show_in_front = True

        # Disabilita il render
        helper.hide_render = True

        # Imposta dimensioni e posizione in base ai confini
        w = bounds.x_max - bounds.x_min
        d = bounds.y_max - bounds.y_min
        h = bounds.z_ceiling - bounds.z_floor

        helper.location = [
            (bounds.x_min + bounds.x_max) / 2.0,
            (bounds.y_min + bounds.y_max) / 2.0,
            (bounds.z_floor + bounds.z_ceiling) / 2.0
        ]
        helper.scale = [w, d, h]

        # Applica scala per facilitare l'editing
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        self.report({"INFO"}, "Helper confini creato. Modificalo per cambiare i confini della stanza.")
        return {"FINISHED"}


class NL2SCENE3D_OT_remove_bounds_helper(Operator):
    """Rimuove il cubo di riferimento dei confini."""

    bl_idname = "nl2scene3d.remove_bounds_helper"
    bl_label = "Rimuovi Helper Confini"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        helper_name = "NL2_RoomBounds_Helper"
        helper = bpy.data.objects.get(helper_name)
        if helper:
            bpy.data.objects.remove(helper, do_unlink=True)
            self.report({"INFO"}, "Helper confini rimosso.")
        else:
            self.report({"WARNING"}, "Nessun helper confini trovato.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Registrazione
# ---------------------------------------------------------------------------

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
    NL2SCENE3D_OT_apply_from_clipboard,
    NL2SCENE3D_OT_apply_from_file,
    NL2SCENE3D_OT_reorganize_with_api,
    NL2SCENE3D_OT_install_pillow,
    NL2SCENE3D_OT_create_bounds_helper,
    NL2SCENE3D_OT_remove_bounds_helper,
)


def register():
    """Registra tutte le classi operatore in Blender."""
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    """Deregistra tutte le classi operatore da Blender."""
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass