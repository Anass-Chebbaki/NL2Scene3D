# nl2scene3d/core/scene_io.py
"""
Unico ponte tra Blender e il nucleo puro.

Due sole responsabilita':
  - extract_scene_state(): legge la scena bpy aperta e costruisce un SceneState
    (classificazione minima fisso/mobile, confini stanza, padri manuali).
  - apply_state():         scrive location e rotation_euler degli oggetti bpy a
    partire da un SceneState. Niente altro: nessun raycast, nessuno snap a terra.

Niente piu' categorie di mobili ne' grouping automatico: cosa e' fisso e chi e'
figlio di chi lo decide l'utente dal pannello. L'automatico stima fissi solo
camera/luci e gli elementi strutturali (per nome), e ricava i confini stanza.

Z non viene MAI modificata: il suo valore e' responsabilita' del file .blend
originale. Se un oggetto "fluttua" nell'originale, mantiene quella quota.
"""

from __future__ import annotations

import logging
from typing import Optional

from .classify import (
    apply_manual_parents,
    compute_room_bounds,
    resolve_classification,
)
from .models import SceneObject, SceneState, Transform
from .settings import CONST, Constants

logger = logging.getLogger(__name__)

_HOME_LOC = "nl2_home_loc"  # posa originale: location locale
_HOME_ROT = "nl2_home_rot"  # posa originale: rotation_euler locale


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
        1. Itera gli oggetti -> classificazione minima (fisso/mobile) + override.
        2. Calcola i RoomBounds (geometrici).
        3. Applica i rapporti padre-figlio MANUALI scelti dall'utente.

    overrides (o None): dict {nome_oggetto: {"fixed": bool, "parent": str}}.
        "fixed"  -> vince sulla stima automatica fisso/mobile.
        "parent" -> nome del padre scelto a mano (stringa vuota = nessun padre).
    """
    try:
        import bpy        # type: ignore  # noqa: PLC0415
        import mathutils  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("bpy/mathutils richiedono l'ambiente Blender.") from exc

    scene          = bpy.context.scene
    effective_name = scene_name or scene.name
    overrides      = overrides or {}

    logger.info("Estraggo scena '%s' (%d oggetti).", effective_name, len(scene.objects))

    objects: list[SceneObject] = []
    movable_count = 0

    for b_obj in scene.objects:
        name       = b_obj.name
        obj_type   = b_obj.type
        dimensions = [b_obj.dimensions.x, b_obj.dimensions.y, b_obj.dimensions.z]

        category, is_movable = resolve_classification(
            name, obj_type, dimensions, overrides.get(name), const
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

    # Parenting MANUALE: la mappa {figlio: padre} arriva dagli override dell'utente.
    parent_map = {
        n: ov["parent"]
        for n, ov in overrides.items()
        if isinstance(ov, dict) and ov.get("parent")
    }
    apply_manual_parents(objects, parent_map)

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

    # Avviso scala: un oggetto che occupa una quota implausibile della stanza e'
    # quasi sempre scala non applicata o unita' d'import sbagliate. Aiuta a capire
    # cosa ridimensionare prima di mandare la scena all'LLM.
    if rb is not None:
        room_w, room_d = rb.x_max - rb.x_min, rb.y_max - rb.y_min
        oversized = []
        for o in state.objects:
            if o.category in ("structural", "technical"):
                continue
            w, d = o.transform.dimensions[0], o.transform.dimensions[1]
            if (room_w > 0 and w > 0.7 * room_w) or (room_d > 0 and d > 0.7 * room_d):
                oversized.append((o.name, w, d))
        if oversized:
            lines.append("")
            lines.append("ATTENZIONE: impronta sospetta (scala/unita'?), ridimensiona questi oggetti:")
            for n, w, d in oversized:
                lines.append(f"  - {n}: {w:.2f} x {d:.2f} m")

    return "\n".join(lines)


def apply_state(state: SceneState, tolerance: float = 0.001) -> dict[str, int]:
    """
    Applica un SceneState alla scena Blender aperta: porta ogni oggetto mobile
    alla posa MONDO indicata (location + rotation_euler). Non aggiunge, rimuove
    o sposta in Z autonomamente.

    Imposta direttamente `matrix_world`: cosi' Blender ricalcola correttamente la
    matrice locale tenendo conto di un eventuale parent nativo E della
    `matrix_parent_inverse` (la matrice che Blender salva quando fai un parenting
    "Keep Transform"). E' il motivo per cui un approccio "world -> local" fatto a
    mano sbaglierebbe la quota dei figli imparentati.

    Gli oggetti vengono processati in ordine di profondita' del parent nativo,
    cosi' un padre nativo e' sempre posizionato prima dei suoi figli.

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

    def native_depth(b_obj) -> int:
        d, p = 0, b_obj.parent
        while p is not None:
            d += 1
            p = p.parent
        return d

    to_process: list[tuple] = []
    for scene_obj in state.objects:
        b_obj = scene.objects.get(scene_obj.name)
        if b_obj is None:
            counters["not_found"] += 1
            continue
        if not scene_obj.is_movable or b_obj.type in ("CAMERA", "LIGHT"):
            counters["skipped"] += 1
            continue
        to_process.append((scene_obj, b_obj))

    # Padri nativi prima dei figli (profondita' crescente).
    to_process.sort(key=lambda pair: native_depth(pair[1]))

    def process_object(scene_obj, b_obj) -> bool:
        t = scene_obj.transform

        cur     = b_obj.matrix_world.translation
        cur_rot = b_obj.matrix_world.to_euler("XYZ")
        moved = (
            any(abs(t.location[i] - cur[i]) > tolerance for i in range(3))
            or any(abs(t.rotation_euler[i] - cur_rot[i]) > tolerance for i in range(3))
        )
        if not moved:
            return False

        b_obj.rotation_mode = "XYZ"
        loc_m = mathutils.Matrix.Translation(t.location)
        rot_m = mathutils.Euler(t.rotation_euler, "XYZ").to_matrix().to_4x4()
        scl   = b_obj.matrix_world.to_scale()
        scl_m = mathutils.Matrix.Diagonal((scl.x, scl.y, scl.z, 1.0))
        # Impostare matrix_world fa gestire a Blender parent + matrix_parent_inverse.
        b_obj.matrix_world = loc_m @ rot_m @ scl_m
        return True

    for scene_obj, b_obj in to_process:
        if process_object(scene_obj, b_obj):
            counters["updated"] += 1
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
        else:
            counters["skipped"] += 1

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    logger.info(
        "Applicazione completa: %d aggiornati, %d non trovati, %d invariati.",
        counters["updated"], counters["not_found"], counters["skipped"],
    )
    return counters