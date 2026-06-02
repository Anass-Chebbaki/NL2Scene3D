# nl2scene3d/core/scene_io.py
"""
Unico ponte tra Blender e il nucleo puro.

Due sole responsabilita':
  - extract_scene_state(): legge la scena bpy aperta e costruisce un SceneState
    (classificazione, confini stanza, grouping, regole statiche).
  - apply_state():         scrive location e rotation_euler degli oggetti bpy a
    partire da un SceneState. Niente altro: nessun raycast, nessuno snap a terra.

La persistenza del grouping (custom property 'nl2_parent') vive QUI, non dentro
compute_grouping: cosi' la logica di grouping resta pura e testabile, e Blender
e' l'unico a sapere come ricordare i gruppi tra una chiamata e l'altra.

Z non viene MAI modificata: il suo valore e' responsabilita' del file .blend
originale. Se un oggetto "fluttua" nell'originale, mantiene quella quota.
"""

from __future__ import annotations

import logging
from typing import Optional

from .classify import (
    apply_static_placement_rules,
    classify_object,
    compute_grouping,
    compute_room_bounds,
    resolve_classification,
)
from .models import SceneObject, SceneState, Transform
from .settings import CONST, Constants

logger = logging.getLogger(__name__)

_GROUP_PROP = "nl2_parent"  # custom property usata per ricordare il grouping
_HOME_LOC   = "nl2_home_loc"  # posa originale: location locale
_HOME_ROT   = "nl2_home_rot"  # posa originale: rotation_euler locale


# ---------------------------------------------------------------------------
# Stato "originale" della scena (per il reset)
# ---------------------------------------------------------------------------

def capture_home_state() -> int:
    """
    Salva la posa corrente (location + rotation locali) di ogni oggetto come
    stato "originale", in custom property che persistono nel .blend.

    Non sovrascrive uno snapshot gia' esistente: cosi' il primo Randomize
    fotografa la scena pristina e i Randomize successivi non rovinano
    l'originale. Ritorna il numero di oggetti fotografati.
    """
    import bpy  # noqa: PLC0415
    n = 0
    for obj in bpy.context.scene.objects:
        if _HOME_LOC in obj:
            continue
        obj[_HOME_LOC] = tuple(obj.location)
        obj[_HOME_ROT] = tuple(obj.rotation_euler)
        n += 1
    return n


def has_home_state() -> bool:
    """True se almeno un oggetto ha uno snapshot 'originale'."""
    import bpy  # noqa: PLC0415
    return any(_HOME_LOC in o for o in bpy.context.scene.objects)


def reset_home_state() -> int:
    """
    Ripristina location + rotation di ogni oggetto dallo snapshot 'originale'.
    Ritorna quanti oggetti sono stati ripristinati.
    """
    import bpy  # noqa: PLC0415
    n = 0
    for obj in bpy.context.scene.objects:
        if _HOME_LOC not in obj:
            continue
        loc = obj[_HOME_LOC]
        obj.location = (loc[0], loc[1], loc[2])
        rot = obj.get(_HOME_ROT)
        if rot is not None:
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = (rot[0], rot[1], rot[2])
        n += 1
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    return n


# ---------------------------------------------------------------------------
# Persistenza del grouping (solo qui si tocca bpy per i gruppi)
# ---------------------------------------------------------------------------

def _read_prior_grouping(objects: list[SceneObject]) -> dict[str, str]:
    """Legge 'nl2_parent' dagli oggetti Blender e ritorna {figlio: padre}."""
    import bpy  # noqa: PLC0415
    names = {o.name for o in objects}
    prior: dict[str, str] = {}
    for o in objects:
        b_obj = bpy.data.objects.get(o.name)
        if b_obj is None:
            continue
        parent = b_obj.get(_GROUP_PROP)
        if parent and parent in names:
            prior[o.name] = parent
    return prior


def _persist_grouping(objects: list[SceneObject]) -> None:
    """Scrive 'nl2_parent' su ogni oggetto Blender (vuoto se senza padre)."""
    import bpy  # noqa: PLC0415
    for o in objects:
        b_obj = bpy.data.objects.get(o.name)
        if b_obj is not None:
            b_obj[_GROUP_PROP] = o.parent if o.parent else ""


# ---------------------------------------------------------------------------
# Estrazione
# ---------------------------------------------------------------------------

def extract_scene_state(
    scene_name: Optional[str] = None,
    overrides: Optional[dict] = None,
    const: Constants = CONST,
) -> SceneState:
    """
    Estrae lo stato completo della scena Blender attualmente aperta.

    Passi:
        1. Itera gli oggetti -> classifica (auto + eventuale override) -> SceneObject.
        2. Calcola i RoomBounds.
        3. Calcola/riusa il grouping padre-figlio e lo annota su ogni oggetto.
        4. Congela mensole/oggetti a soffitto, rispettando gli override "mobile".

    overrides (o None): dict {nome_oggetto: {"fixed": bool, "category": str}}.
    Se fornito, vince sulla classificazione automatica per quegli oggetti.
    """
    try:
        import bpy        # type: ignore  # noqa: PLC0415
        import mathutils  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("bpy/mathutils richiedono l'ambiente Blender.") from exc

    scene          = bpy.context.scene
    effective_name = scene_name or scene.name

    logger.info("Estraggo scena '%s' (%d oggetti).", effective_name, len(scene.objects))

    objects: list[SceneObject] = []
    movable_count = 0

    for b_obj in scene.objects:
        name       = b_obj.name
        obj_type   = b_obj.type
        dimensions = [b_obj.dimensions.x, b_obj.dimensions.y, b_obj.dimensions.z]

        category, is_movable = resolve_classification(
            name, obj_type, dimensions, (overrides or {}).get(name), const
        )

        # Limite di oggetti mobili.
        if is_movable and movable_count >= const.max_movable_objects:
            is_movable = False

        # Forza una modalita' di rotazione Euler compatibile.
        if b_obj.rotation_mode not in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
            b_obj.rotation_mode = "XYZ"

        # Offset dell'origine rispetto al centro geometrico (m, gia' scalato).
        local_center = (
            sum((mathutils.Vector(p) for p in b_obj.bound_box), mathutils.Vector()) / 8
        )
        origin_offset = [
            local_center.x * b_obj.scale.x,
            local_center.y * b_obj.scale.y,
            local_center.z * b_obj.scale.z,
        ]

        # Coordinate mondo (matrix_world), sempre, anche con un parent Blender nativo.
        world_mat = b_obj.matrix_world
        world_loc = world_mat.translation
        world_rot = world_mat.to_euler("XYZ")

        transform = Transform(
            location=[world_loc.x, world_loc.y, world_loc.z],
            rotation_euler=[world_rot.x, world_rot.y, world_rot.z],
            dimensions=dimensions,
            origin_offset=origin_offset,
        )

        objects.append(SceneObject(
            name=name, object_type=obj_type, transform=transform,
            category=category, is_movable=is_movable,
        ))
        if is_movable:
            movable_count += 1

    logger.info("Estratti %d oggetti (%d mobili).", len(objects), movable_count)

    room_bounds = compute_room_bounds(objects)

    # Grouping: riusa quello memorizzato se presente, altrimenti lo calcola e lo salva.
    prior = _read_prior_grouping(objects)
    compute_grouping(objects, prior=prior)
    _persist_grouping(objects)

    # Oggetti dichiarati esplicitamente MOBILI dall'utente: non vanno congelati.
    protected = {
        name for name, ov in (overrides or {}).items()
        if isinstance(ov, dict) and ov.get("fixed") is False
    }
    apply_static_placement_rules(objects, room_bounds, const, protected=protected)

    return SceneState(
        scene_name=effective_name,
        objects=objects,
        room_bounds=room_bounds,
        pipeline_step="original",
    )


# ---------------------------------------------------------------------------
# Applicazione
# ---------------------------------------------------------------------------

def format_inspection(state: SceneState) -> str:
    """
    Costruisce un report testuale leggibile della scena: per ogni oggetto
    categoria, stato (mobile/fisso) e padre. PURO: nessun bpy. Usato dal
    dry-run per far vedere COME viene interpretata la scena prima di muovere
    qualcosa, cosi' si individuano subito eventuali nomi classificati male.
    """
    rb = state.room_bounds
    movable = [o for o in state.objects if o.is_movable]
    fixed   = [o for o in state.objects if not o.is_movable]
    groups  = [o for o in state.objects if o.children]

    lines: list[str] = []
    lines.append("NL2Scene3D - Inspect (dry-run)")
    lines.append(f"Scena: {state.scene_name}")
    if rb is not None:
        lines.append(
            f"Confini: X[{rb.x_min:.2f}, {rb.x_max:.2f}]  "
            f"Y[{rb.y_min:.2f}, {rb.y_max:.2f}]  "
            f"Z[{rb.z_floor:.2f}, {rb.z_ceiling:.2f}]"
        )
    lines.append(
        f"Oggetti: {len(state.objects)} "
        f"(mobili: {len(movable)}, fissi: {len(fixed)}), gruppi: {len(groups)}"
    )
    lines.append("")
    lines.append(f"{'NOME':<26}{'CATEGORIA':<18}{'STATO':<9}PADRE")
    lines.append("-" * 70)
    for o in state.objects:
        stato = "mobile" if o.is_movable else "FISSO"
        lines.append(f"{o.name:<26}{o.category:<18}{stato:<9}{o.parent or '-'}")

    # Avviso: oggetti MOBILI il cui nome contiene un termine "strutturale".
    # Sono i candidati a una classificazione sbagliata (verra' spostato un muro?).
    struct_kw = (
        "door", "window", "wall", "ceiling",
        "porta", "finestra", "muro", "parete", "soffitto",
    )
    suspicious = [o.name for o in movable if any(k in o.name.lower() for k in struct_kw)]
    if suspicious:
        lines.append("")
        lines.append("ATTENZIONE: oggetti MOBILI con nome 'strutturale' (controlla):")
        for n in suspicious:
            lines.append(f"  - {n}")

    return "\n".join(lines)


def apply_state(state: SceneState, tolerance: float = 0.001) -> dict[str, int]:
    """
    Applica un SceneState alla scena Blender aperta: aggiorna location e
    rotation_euler degli oggetti che corrispondono per nome. Non aggiunge,
    rimuove o sposta in Z autonomamente.

    Ritorna i contatori {'updated', 'not_found', 'skipped'}.
    """
    try:
        import bpy        # noqa: PLC0415
        import mathutils  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("bpy richiede l'ambiente Blender.") from exc

    counters = {"updated": 0, "not_found": 0, "skipped": 0}
    scene = bpy.context.scene

    logger.info(
        "Applico stato '%s' (step: %s, %d oggetti).",
        state.scene_name, state.pipeline_step, len(state.objects),
    )

    roots_to_process: list[tuple] = []
    children_to_process: list[tuple] = []

    for scene_obj in state.objects:
        b_obj = scene.objects.get(scene_obj.name)
        if b_obj is None:
            counters["not_found"] += 1
            continue
        if not scene_obj.is_movable or b_obj.type in ("CAMERA", "LIGHT"):
            counters["skipped"] += 1
            continue
        (roots_to_process if b_obj.parent is None else children_to_process).append(
            (scene_obj, b_obj)
        )

    def process_object(scene_obj, b_obj) -> bool:
        import mathutils  # noqa: PLC0415
        t = scene_obj.transform
        updated = False

        # --- Location ---
        cur = b_obj.matrix_world.translation
        cur_loc = [cur.x, cur.y, cur.z]
        if any(abs(t.location[i] - cur_loc[i]) > tolerance for i in range(3)):
            if b_obj.parent is not None:
                try:
                    world_vec = mathutils.Vector(t.location)
                    local_vec = b_obj.parent.matrix_world.inverted() @ world_vec
                    b_obj.location = (local_vec.x, local_vec.y, local_vec.z)
                except Exception:
                    b_obj.location = (t.location[0], t.location[1], t.location[2])
            else:
                b_obj.location = (t.location[0], t.location[1], t.location[2])
            updated = True

        # --- Rotation ---
        cur_rot = b_obj.matrix_world.to_euler("XYZ")
        if any(abs(t.rotation_euler[i] - cur_rot[i]) > tolerance for i in range(3)):
            b_obj.rotation_mode = "XYZ"
            if b_obj.parent is not None:
                try:
                    world_rot = mathutils.Euler(t.rotation_euler, "XYZ")
                    local_mat = (
                        b_obj.parent.matrix_world.to_3x3().inverted()
                        @ world_rot.to_matrix()
                    )
                    local_rot = local_mat.to_euler("XYZ")
                    b_obj.rotation_euler = (local_rot.x, local_rot.y, local_rot.z)
                except Exception:
                    b_obj.rotation_euler = tuple(t.rotation_euler)
            else:
                b_obj.rotation_euler = tuple(t.rotation_euler)
            updated = True

        return updated

    # Pass 1: root.
    for scene_obj, b_obj in roots_to_process:
        counters["updated" if process_object(scene_obj, b_obj) else "skipped"] += 1

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    # Pass 2: figli (aggiorna le matrici tra una scrittura e l'altra).
    for scene_obj, b_obj in children_to_process:
        upd = process_object(scene_obj, b_obj)
        counters["updated" if upd else "skipped"] += 1
        if upd:
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    logger.info(
        "Applicazione completa: %d aggiornati, %d non trovati, %d invariati.",
        counters["updated"], counters["not_found"], counters["skipped"],
    )
    return counters
