# nl2scene3d/scene_state.py
"""
Caricamento e costruzione dello SceneState da una scena Blender aperta.

Responsabilità di questo modulo:
1. Estrarre oggetti e trasformazioni dalla scena bpy corrente.
2. Classificare ogni oggetto (categoria + is_movable).
3. Calcolare i RoomBounds dalla geometria strutturale.
4. Rilevare le relazioni parent-child (grouping) e annotarle
   direttamente su ogni SceneObject (campo parent/children).
   Il grouping viene calcolato UNA SOLA VOLTA qui, non in ogni step.
5. Serializzare/deserializzare SceneState da/verso JSON.

Non contiene logica di randomizzazione, LLM o rendering.
Deve essere eseguito all'interno dell'ambiente Python di Blender.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from nl2scene3d.config import PipelineConfig
from nl2scene3d.models import RoomBounds, SceneObject, SceneState, Transform

try:
    import bpy  # type: ignore
except ImportError:
    bpy = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classificazione oggetti
# ---------------------------------------------------------------------------

def _has_kw(keywords: list[str] | tuple[str, ...], text: str) -> bool:
    return any(k in text for k in keywords)


def classify_object(
    name: str,
    object_type: str,
    dimensions: list[float],
    config: PipelineConfig,
) -> tuple[str, bool]:
    """
    Determina (categoria, is_movable) di un oggetto.

    Returns:
        (categoria, is_movable)
    """
    name_lower = name.lower()

    # Tipi non-mesh → sempre statici
    if object_type in config.non_mesh_types:
        return "technical", False

    # Oggetti troppo piccoli → decorazioni fisse (pomelli, viti, ecc.)
    max_dim = max(dimensions) if dimensions else 0.0
    if max_dim < config.min_object_dimension:
        return "decoration_small", False

    # Luci
    if _has_kw(("lamp", "lampada", "light"), name_lower):
        if _has_kw(config.ceiling_light_patterns, name_lower):
            return "light_ceiling", False
        return "light_floor", True

    # Pomelli e maniglie → fissi
    if _has_kw(("knob", "pomello", "handle", "maniglia"), name_lower):
        return "technical", False

    # Decorazioni / elettronica da scrivania
    if _has_kw(
        ("decor", "decoration", "ornament", "book", "bottle",
         "monitor", "pc", "computer", "keyboard", "mouse", "trashbin"),
        name_lower,
    ):
        return "decoration", True

    # Strutturali → sempre statici (controllo dopo le decorazioni per precedenza corretta)
    if _has_kw(config.structural_patterns, name_lower):
        return "structural", False

    # Mobili principali
    if _has_kw(("sofa", "couch", "divano"), name_lower):
        return "seating_large", True
    if _has_kw(("chair", "sedia", "stool", "sgabello"), name_lower):
        return "seating_small", True
    if _has_kw(("table", "tavolo", "desk", "scrivania"), name_lower):
        return "table", True
    if _has_kw(("shelf", "scaffale", "bookcase", "libreria"), name_lower):
        return "storage", True
    if _has_kw(("bed", "letto", "mattress", "materasso"), name_lower):
        return "bed", True
    if _has_kw(("wardrobe", "armadio", "cabinet", "dresser"), name_lower):
        return "storage", True
    if _has_kw(("rug", "tappeto", "carpet"), name_lower):
        return "rug", True
    if _has_kw(("plant", "pianta", "vase", "vaso"), name_lower):
        return "decoration", True

    return "furniture", True


# ---------------------------------------------------------------------------
# Calcolo RoomBounds
# ---------------------------------------------------------------------------

def compute_room_bounds(objects: list[SceneObject]) -> RoomBounds:
    """
    Calcola i bounds della stanza dagli oggetti strutturali.

    Strategia:
    1. Se esiste un'unica mesh-stanza predominante (volume > 50% del totale
       strutturale), usa le sue dimensioni come stanza.
    2. Altrimenti combina gli AABB di tutti gli oggetti strutturali.
    3. La z_ceiling viene ricavata dagli oggetti 'ceiling/room/roof';
       se assente, si usa il massimo Z strutturale o 2.5m come fallback.
    """
    structural = [o for o in objects if o.category == "structural"]
    if not structural:
        logger.warning("Nessun oggetto strutturale trovato. Uso l'insieme completo degli oggetti.")
        structural = objects
    if not structural:
        logger.warning("Scena vuota — bounds di default ±5m.")
        return RoomBounds(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0)

    # z_ceiling dinamica
    ceiling_kws = ("ceiling", "room", "roof", "soffitto")
    ceiling_objs = [o for o in structural if _has_kw(ceiling_kws, o.name.lower())]
    if ceiling_objs:
        z_ceiling = max(
            o.transform.z_range()[1]
            for o in ceiling_objs
        )
    else:
        max_z = max(
            (o.transform.z_range()[1] for o in structural),
            default=2.5,
        )
        z_ceiling = max_z if max_z > 1.0 else 2.5

    # Strategia 1: mesh-stanza unica
    vols = [
        (o, o.transform.dimensions[0] * o.transform.dimensions[1] * o.transform.dimensions[2])
        for o in structural
    ]
    largest_obj, max_vol = max(vols, key=lambda x: x[1])
    total_vol = sum(v for _, v in vols)
    if total_vol > 0 and max_vol > 0.5 * total_vol and max_vol > 1.0:
        x_min, x_max, y_min, y_max = largest_obj.transform.aabb_xy(margin=0.0)
        logger.info(
            "Stanza identificata da oggetto unico '%s' (AABB: X[%.2f, %.2f] Y[%.2f, %.2f]).",
            largest_obj.name, x_min, x_max, y_min, y_max
        )
        return RoomBounds(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_floor=0.0,
            z_ceiling=z_ceiling,
        )

    # Strategia 2: AABB unione di tutti i strutturali
    aabbs = [o.transform.aabb_xy(margin=0.0) for o in structural]
    return RoomBounds(
        x_min=min(a[0] for a in aabbs),
        x_max=max(a[1] for a in aabbs),
        y_min=min(a[2] for a in aabbs),
        y_max=max(a[3] for a in aabbs),
        z_floor=0.0,
        z_ceiling=z_ceiling,
    )


# ---------------------------------------------------------------------------
# Grouping (parent-child) — calcolato una volta sola
# ---------------------------------------------------------------------------

def _volume(o: SceneObject) -> float:
    d = o.transform.dimensions
    return d[0] * d[1] * d[2]


def _sat_overlap(poly_a: list[tuple[float, float]], poly_b: list[tuple[float, float]]) -> bool:
    """Separating Axis Theorem per due poligoni convessi 2D."""
    def axes(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        result = []
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            mag = math.hypot(ex, ey)
            if mag > 1e-6:
                result.append((-ey / mag, ex / mag))
        return result

    def project(poly: list[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(dots), max(dots)

    for axis in axes(poly_a) + axes(poly_b):
        mn_a, mx_a = project(poly_a, axis)
        mn_b, mx_b = project(poly_b, axis)
        if mx_a < mn_b or mx_b < mn_a:
            return False  # Asse separante trovato
    return True  # Nessun asse separante → collisione


def compute_grouping(objects: list[SceneObject]) -> None:
    """
    Rileva le relazioni parent-child e le annota in-place su ogni SceneObject.

    Un oggetto B è figlio di A se UNA delle seguenti condizioni è vera:
    1. Superficie: B è appoggiato sopra A  (z_bottom di B ≈ z_top di A, ±15cm)
                   E i footprint XY si sovrappongono.
    2. Contenuto:  B è contenuto nel range Z di A (es. libro in scaffale)
                   E i footprint XY si sovrappongono.
    3. Prossimità: i range Z di A e B si sovrappongono significativamente
                   (≥30% dell'altezza di B) E i footprint XY sono entro 60cm
                   (es. sedia sotto scrivania, PC tower a lato della scrivania).

    In tutti i casi A deve avere volume ≥ 1.5× quello di B.

    Ogni oggetto riceve:
    - obj.parent: nome del parent diretto (o None)
    - obj.children: lista dei figli diretti

    Il grouping è diretto (padre-figlio), non transitivo.
    """
    # Reset local
    for obj in objects:
        obj.parent = None
        obj.children = []

    movable = [o for o in objects if o.is_movable and o.category != "structural"]
    by_name = {o.name: o for o in objects}

    # 1. Tenta di caricare il grouping esistente dalle proprietà Custom di Blender
    #    Questo garantisce che il raggruppamento sia "stabile" durante la sessione.
    has_custom_props = False
    if bpy:
        for obj in objects:
            # bpy.data.objects è una collezione Blender valida, ma il linter non la riconosce
            if obj.name in bpy.data.objects:  # type: ignore[operator]
                b_obj = bpy.data.objects[obj.name]  # type: ignore[index]
                p_name = b_obj.get("nl2_parent")
                if p_name and p_name in by_name:
                    obj.parent = p_name
                    by_name[p_name].children.append(obj.name)
                    has_custom_props = True

    if has_custom_props:
        logger.info("Grouping persistente caricato dalle proprietà di Blender.")
        return

    # 2. Se non ci sono proprietà salvate, calcola il grouping originale
    # Definiamo rigorosamente chi può essere figlio e chi può essere padre
    # Mobili principali (Letti, Tavoli, Armadi) devono essere sempre ROOT indipendenti.
    ALLOWED_CHILD_CATEGORIES = {"decoration", "decoration_small", "seating_small", "light_floor"}
    ALLOWED_PARENT_CATEGORIES = {"table", "desk", "storage", "seating_large", "bed", "furniture"}

    for child in movable:
        # SE L'OGGETTO È UN MOBILE PRINCIPALE, NON PUÒ ESSERE FIGLIO DI NESSUNO
        if child.category not in ALLOWED_CHILD_CATEGORIES:
            continue

        child_z_min, child_z_max = child.transform.z_range()
        child_vol = _volume(child)
        child_height = child_z_max - child_z_min
        child_poly = child.transform.obb_corners_xy(margin=0.0)

        best_parent: Optional[str] = None
        best_score: float = float("inf")   # punteggio basso = parent migliore

        for candidate in movable:
            if candidate.name == child.name:
                continue
            
            # SOLO LE CATEGORIE PADRE POSSONO AVERE FIGLI
            if candidate.category not in ALLOWED_PARENT_CATEGORIES:
                continue

            cand_vol = _volume(candidate)
            if cand_vol < child_vol * 1.2:
                continue  # Il parent deve essere almeno un po' più grande

            par_z_min, par_z_max = candidate.transform.z_range()

            # --- Criterio 1: appoggiato sopra ---
            z_diff_top = child_z_min - par_z_max
            is_on_top = -0.05 <= z_diff_top <= 0.15   # sopra entro 15cm, nega falsi positivi sopra 5cm

            # --- Criterio 2: contenuto nel range Z del parent ---
            is_inside = (
                child_z_min >= par_z_min - 0.05
                and child_z_max <= par_z_max + 0.05
            )

            # --- Criterio 3: Z-overlap + prossimità XY (sedia sotto scrivania, PC a lato) ---
            z_overlap = max(0.0, min(child_z_max, par_z_max) - max(child_z_min, par_z_min))
            has_z_overlap = (child_height > 0) and (z_overlap / child_height >= 0.30)

            matched = False
            score = 0.0

            if is_on_top or is_inside:
                # Check sovrapposizione XY diretta
                par_poly = candidate.transform.obb_corners_xy(margin=0.0)
                if _sat_overlap(child_poly, par_poly):
                    matched = True
                    score = abs(z_diff_top) if is_on_top else 0.0

            if not matched and has_z_overlap:
                # Check prossimità XY ridotta (25cm) e solo per categorie MINORI
                # Solo sedie e decorazioni possono seguire per prossimità.
                # Mobili come comodini o cassettiere rimangono indipendenti.
                allowed_proximity_children = {"seating_small", "decoration", "decoration_small", "light_floor"}
                allowed_proximity_parents = {"table", "desk", "storage", "seating_large", "bed", "furniture"}

                if (child.category in allowed_proximity_children and 
                    candidate.category in allowed_proximity_parents):
                    
                    par_poly_expanded = candidate.transform.obb_corners_xy(margin=0.15)
                    if _sat_overlap(child_poly, par_poly_expanded):
                        matched = True
                        # Punteggio = distanza dal centro del parent in XY
                        cx, cy = candidate.transform.geometric_center_xy()
                        bx, by = child.transform.geometric_center_xy()
                        score = 10.0 + math.hypot(bx - cx, by - cy)

            if not matched:
                continue

            # Scegli il parent con punteggio più basso (più vicino / più sopra)
            if score < best_score:
                best_score = score
                best_parent = candidate.name

        if best_parent is not None:
            child.parent = best_parent
            by_name[best_parent].children.append(child.name)
            logger.debug("Grouping: '%s' → parent '%s' (score=%.2f).", child.name, best_parent, best_score)

    # 3. Salva il grouping calcolato nelle proprietà di Blender per le prossime chiamate
    if bpy:
        for obj in objects:
            if obj.name in bpy.data.objects:  # type: ignore[operator]
                b_obj = bpy.data.objects[obj.name]  # type: ignore[index]
                # Salviamo il nome del parent (stringa vuota se è un root)
                b_obj["nl2_parent"] = obj.parent if obj.parent else ""

    n_groups = sum(1 for o in objects if o.children)
    n_children = sum(1 for o in objects if o.parent is not None)
    logger.info("Grouping completato: %d gruppi, %d oggetti figlio.", n_groups, n_children)


# ---------------------------------------------------------------------------
# SceneLoader
# ---------------------------------------------------------------------------

class SceneLoader:
    """
    Carica e ispeziona una scena Blender, producendo uno SceneState completo.

    Deve essere usato all'interno dell'ambiente Python di Blender (bpy disponibile).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        logger.info(
            "SceneLoader inizializzato. Max oggetti movibili: %d.", config.max_movable_objects
        )

    def load_blend_file(self, blend_path: Path) -> None:
        """Apre un file .blend sostituendo la scena corrente."""
        if not blend_path.exists():
            raise FileNotFoundError(f"File .blend non trovato: {blend_path}")
        try:
            import bpy  # type: ignore # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("bpy richiede l'ambiente Blender.") from exc
        logger.info("Apertura '%s'…", blend_path)
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        logger.info("File .blend aperto.")

    def extract_scene_state(self, scene_name: Optional[str] = None) -> SceneState:
        """
        Estrae lo stato completo della scena Blender corrente.

        Passi:
        1. Itera tutti gli oggetti → classifica → costruisce SceneObject.
        2. Calcola RoomBounds.
        3. Calcola il grouping parent-child e lo annota sugli oggetti.

        Returns:
            SceneState completo con pipeline_step='original'.
        """
        try:
            import bpy        # type: ignore # noqa: PLC0415
            import mathutils  # type: ignore # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("bpy/mathutils richiedono l'ambiente Blender.") from exc

        blender_scene = bpy.context.scene  # type: ignore[union-attr]
        effective_name = scene_name or blender_scene.name  # type: ignore[union-attr]

        logger.info(
            "Estrazione scena '%s' (%d oggetti Blender).",
            effective_name, len(blender_scene.objects),  # type: ignore[union-attr]
        )

        objects: list[SceneObject] = []
        movable_count = 0

        for blender_obj in blender_scene.objects:  # type: ignore[union-attr]
            name = blender_obj.name
            obj_type = blender_obj.type
            dimensions = [
                blender_obj.dimensions.x,
                blender_obj.dimensions.y,
                blender_obj.dimensions.z,
            ]

            category, is_movable = classify_object(name, obj_type, dimensions, self.config)

            # Rispetta il limite di oggetti movibili
            if is_movable and movable_count >= self.config.max_movable_objects:
                logger.debug(
                    "'%s' declassato: limite %d oggetti movibili raggiunto.",
                    name, self.config.max_movable_objects,
                )
                is_movable = False

            # Forza rotazione Euler per compatibilità
            if blender_obj.rotation_mode not in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
                blender_obj.rotation_mode = "XYZ"

            # Offset dell'origine rispetto al centro geometrico (in metri, già scalato)
            local_center = (
                sum((mathutils.Vector(p) for p in blender_obj.bound_box), mathutils.Vector()) / 8
            )
            origin_offset = [
                local_center.x * blender_obj.scale.x,
                local_center.y * blender_obj.scale.y,
                local_center.z * blender_obj.scale.z,
            ]

            # Usa sempre coordinate MONDO (matrix_world) per garantire coerenza
            # anche quando l'oggetto ha un parent nativo in Blender.
            world_mat = blender_obj.matrix_world
            world_loc = world_mat.translation
            world_rot = world_mat.to_euler('XYZ')

            transform = Transform(
                location=[world_loc.x, world_loc.y, world_loc.z],
                rotation_euler=[world_rot.x, world_rot.y, world_rot.z],
                dimensions=dimensions,
                origin_offset=origin_offset,
            )

            scene_obj = SceneObject(
                name=name,
                object_type=obj_type,
                transform=transform,
                category=category,
                is_movable=is_movable,
            )
            objects.append(scene_obj)
            if is_movable:
                movable_count += 1

        logger.info("Estratti %d oggetti (%d movibili).", len(objects), movable_count)

        # Calcola bounds
        room_bounds = compute_room_bounds(objects)
        logger.info(
            "Bounds: X[%.2f, %.2f] Y[%.2f, %.2f] Z[%.2f, %.2f].",
            room_bounds.x_min, room_bounds.x_max,
            room_bounds.y_min, room_bounds.y_max,
            room_bounds.z_floor, room_bounds.z_ceiling,
        )

        # Calcola grouping — annota parent/children su ogni oggetto
        compute_grouping(objects)

        return SceneState(
            scene_name=effective_name,
            objects=objects,
            room_bounds=room_bounds,
            pipeline_step="original",
        )

    # --- Serializzazione ---

    def save_state(self, state: SceneState, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Stato '%s' (step: %s) salvato in: %s", state.scene_name, state.pipeline_step, output_path)

    @staticmethod
    def load_state(json_path: Path) -> SceneState:
        if not json_path.exists():
            raise FileNotFoundError(f"File JSON non trovato: {json_path}")
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        state = SceneState.from_dict(data)
        logger.info("Stato '%s' (step: %s) caricato da: %s", state.scene_name, state.pipeline_step, json_path)
        return state