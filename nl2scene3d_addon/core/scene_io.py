# nl2scene3d/core/scene_io.py
"""
Unico ponte tra Blender e il nucleo puro del progetto.

Il modulo ha due sole responsabilita':

    - extract_scene_state(): legge la scena Blender attualmente aperta e
      costruisce un SceneState con la classificazione minima fisso/mobile,
      i confini della stanza e i rapporti padre-figlio manuali.

    - apply_state(): scrive la location e la rotation_euler degli oggetti
      Blender a partire da un SceneState. Non aggiunge, rimuove o modifica
      la coordinata Z autonomamente.

Classificazione automatica:
    Camera e luci vengono sempre classificate come fisse. Gli elementi
    strutturali vengono identificati per nome. Tutto il resto e' trattato
    come mobile, salvo override espliciti dell'utente.

    I rapporti padre-figlio sono interamente decisi dall'utente tramite il
    pannello; non viene eseguito alcun grouping automatico.

Invarianza di Z:
    La coordinata Z non viene mai modificata da questo modulo. Il suo valore
    e' di esclusiva responsabilita' del file .blend originale. Se un oggetto
    "fluttua" nell'originale, manterra' quella quota anche dopo la
    riorganizzazione.
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

# Chiavi delle custom property usate per memorizzare la posa originale.
_HOME_LOC = "nl2_home_loc"  # Location locale originale.
_HOME_ROT = "nl2_home_rot"  # Rotation Euler locale originale.


# ---------------------------------------------------------------------------
# Gestione dello stato "originale" della scena (per il reset)
# ---------------------------------------------------------------------------

def capture_home_state() -> int:
    """
    Salva la posa corrente (location e rotation_euler locali) di ogni oggetto
    come "stato originale", memorizzandola in custom property che persistono
    nel file .blend.

    Non sovrascrive uno snapshot gia' esistente: il primo Randomize fotografa
    la scena pristina e le esecuzioni successive non alterano l'originale.

    Returns:
        Il numero di oggetti di cui e' stata salvata la posa.
    """
    import bpy  # noqa: PLC0415

    n = 0
    for obj in bpy.context.scene.objects:
        if _HOME_LOC in obj:
            continue  # Snapshot gia' presente: non sovrascrivere.
        obj[_HOME_LOC] = tuple(obj.location)
        obj[_HOME_ROT] = tuple(obj.rotation_euler)
        n += 1

    return n


def has_home_state() -> bool:
    """
    Verifica se esiste almeno uno snapshot "originale" nella scena corrente.

    Returns:
        True se almeno un oggetto ha la custom property dello snapshot.
    """
    import bpy  # noqa: PLC0415
    return any(_HOME_LOC in o for o in bpy.context.scene.objects)


def reset_home_state() -> int:
    """
    Ripristina la location e la rotation_euler di ogni oggetto allo snapshot
    "originale" memorizzato nelle custom property.

    Returns:
        Il numero di oggetti ripristinati.
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
# Estrazione dello stato della scena
# ---------------------------------------------------------------------------

def extract_scene_state(
    scene_name: Optional[str] = None,
    overrides: Optional[dict] = None,
    const: Constants = CONST,
) -> SceneState:
    """
    Estrae lo stato completo della scena Blender attualmente aperta e lo
    restituisce come SceneState.

    Passi eseguiti:
        1. Itera tutti gli oggetti della scena e ne determina la classificazione
           (fisso/mobile, categoria) tramite resolve_classification, tenendo
           conto degli override manuali dell'utente.
        2. Calcola i RoomBounds geometrici dalla geometria degli elementi
           strutturali.
        3. Applica i rapporti padre-figlio manuali specificati negli override.

    Args:
        scene_name: Nome da assegnare alla scena nel SceneState. Se None,
                    viene usato il nome della scena Blender corrente.
        overrides:  Dizionario opzionale {nome_oggetto: {"fixed": bool,
                    "parent": str}}. Il campo "fixed" sovrascrive la stima
                    automatica; il campo "parent" specifica il padre manuale
                    (stringa vuota = nessun padre).
        const:      Costanti di configurazione del progetto.

    Returns:
        Un SceneState con pipeline_step="original".

    Raises:
        ImportError: Se bpy o mathutils non sono disponibili (fuori da Blender).
    """
    try:
        import bpy        # type: ignore  # noqa: PLC0415
        import mathutils  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("bpy/mathutils richiedono l'ambiente Blender.") from exc

    scene = bpy.context.scene
    effective_name = scene_name or scene.name
    overrides = overrides or {}

    logger.info("Estraggo scena '%s' (%d oggetti).", effective_name, len(scene.objects))

    objects: list[SceneObject] = []
    movable_count = 0

    for b_obj in scene.objects:
        name = b_obj.name
        obj_type = b_obj.type
        dimensions = [b_obj.dimensions.x, b_obj.dimensions.y, b_obj.dimensions.z]

        category, is_movable = resolve_classification(
            name, obj_type, dimensions, overrides.get(name), const
        )

        # Rispetta il limite massimo di oggetti mobili.
        if is_movable and movable_count >= const.max_movable_objects:
            is_movable = False

        # Assicura una modalita' di rotazione Euler compatibile con XYZ.
        if b_obj.rotation_mode not in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
            b_obj.rotation_mode = "XYZ"

        # Offset dell'origine rispetto al centro geometrico della mesh (in metri,
        # gia' moltiplicato per la scala dell'oggetto).
        local_center = (
            sum((mathutils.Vector(p) for p in b_obj.bound_box), mathutils.Vector()) / 8
        )
        origin_offset = [
            local_center.x * b_obj.scale.x,
            local_center.y * b_obj.scale.y,
            local_center.z * b_obj.scale.z,
        ]

        # Le coordinate vengono sempre lette da matrix_world, che e' corretta
        # anche in presenza di un parent nativo Blender.
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
            name=name,
            object_type=obj_type,
            transform=transform,
            category=category,
            is_movable=is_movable,
        ))

        if is_movable:
            movable_count += 1

    logger.info("Estratti %d oggetti (%d mobili).", len(objects), movable_count)

    room_bounds = compute_room_bounds(objects)

    # Costruisce la mappa padre-figlio dai soli override che specificano un padre.
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
# Report di ispezione (dry-run, puro)
# ---------------------------------------------------------------------------

def format_inspection(state: SceneState) -> str:
    """
    Costruisce un report testuale leggibile dello stato della scena, utile
    per il dry-run: mostra come la scena viene interpretata (categoria, stato
    fisso/mobile, padre) prima di eseguire qualsiasi operazione.

    Include avvisi per:
    - Oggetti MOBILI il cui nome contiene termini strutturali (potenziale
      classificazione errata: e' possibile che si sposti un muro?).
    - Oggetti con impronta XY sospettamente grande rispetto alla stanza
      (probabile scala non applicata o unita' di import errate).

    Funzione pura: non accede a bpy.

    Args:
        state: Lo stato della scena da ispezionare.

    Returns:
        Stringa multiriga con il report formattato.
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
            f"Confini: X[{rb.x_min:.2f}, {rb.x_max:.2f}] "
            f"Y[{rb.y_min:.2f}, {rb.y_max:.2f}] "
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

    # Avviso: oggetti mobili con nomi che suggeriscono elementi strutturali.
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

    # Avviso: oggetti con impronta sospettamente grande rispetto alla stanza.
    if rb is not None:
        room_w = rb.x_max - rb.x_min
        room_d = rb.y_max - rb.y_min
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


# ---------------------------------------------------------------------------
# Applicazione dello stato alla scena Blender
# ---------------------------------------------------------------------------

def apply_state(state: SceneState, tolerance: float = 0.001) -> dict[str, int]:
    """
    Applica un SceneState alla scena Blender aperta, portando ogni oggetto
    mobile alla posa in spazio mondo indicata (location + rotation_euler).
    Non aggiunge, rimuove ne' modifica la coordinata Z autonomamente.

    La posa viene impostata tramite matrix_world, in modo che Blender
    ricalcoli correttamente la matrice locale tenendo conto di un eventuale
    parent nativo e della matrix_parent_inverse (la matrice salvata da
    Blender al momento del parenting "Keep Transform"). Un approccio basato
    su world -> local calcolato a mano sbaglierebbe la quota dei figli
    imparentati.

    Gli oggetti vengono processati in ordine di profondita' del parent nativo
    (prima i padri, poi i figli), garantendo che la posizione del padre sia
    sempre aggiornata prima di quella del figlio.

    Args:
        state:     Lo SceneState con le nuove pose da applicare.
        tolerance: Soglia di spostamento minimo sotto la quale un oggetto non
                   viene aggiornato (evita scritture inutili).

    Returns:
        Dizionario con i contatori {'updated', 'not_found', 'skipped'}.

    Raises:
        ImportError: Se bpy non e' disponibile (fuori da Blender).
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
        """Calcola la profondita' del parent nativo Blender di un oggetto."""
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

    # Ordina per profondita' crescente: i padri nativi vengono processati prima.
    to_process.sort(key=lambda pair: native_depth(pair[1]))

    def process_object(scene_obj: SceneObject, b_obj) -> bool:
        """
        Aggiorna la matrix_world dell'oggetto Blender se la nuova posa
        differisce da quella corrente oltre la soglia di tolleranza.

        Returns:
            True se l'oggetto e' stato aggiornato, False se era gia' in posa.
        """
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

        # L'assegnazione a matrix_world delega a Blender la gestione del parent
        # nativo e della matrix_parent_inverse.
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


# ---------------------------------------------------------------------------
# Metriche di spostamento (snapshot in spazio mondo)
#
# Confronto: originale (O) -> randomizzato (R) -> riordino corrente (C).
# ---------------------------------------------------------------------------

def capture_pose_snapshot(tag: str) -> int:
    """
    Salva la posa in spazio mondo (location + rotation Euler XYZ) di ogni
    oggetto MESH come custom property nl2_{tag}_loc / nl2_{tag}_rot.
    Viene usata per calcolare le metriche di spostamento.

    Args:
        tag: Etichetta dello snapshot (es. "m_orig", "m_rand").

    Returns:
        Il numero di oggetti di cui e' stata salvata la posa.
    """
    import bpy  # noqa: PLC0415

    kl = f"nl2_{tag}_loc"
    kr = f"nl2_{tag}_rot"
    n = 0

    for obj in bpy.context.scene.objects:
        if obj.type in {"CAMERA", "LIGHT"}:
            continue
        m = obj.matrix_world
        obj[kl] = tuple(m.translation)
        obj[kr] = tuple(m.to_euler("XYZ"))
        n += 1

    return n


def _read_snapshot(tag: str) -> dict:
    """
    Legge lo snapshot di posa identificato da 'tag' dalle custom property
    degli oggetti Blender.

    Args:
        tag: Etichetta dello snapshot da leggere.

    Returns:
        Dizionario {nome_oggetto: (loc_tuple, rot_tuple)}.
    """
    import bpy  # noqa: PLC0415

    kl = f"nl2_{tag}_loc"
    kr = f"nl2_{tag}_rot"
    out = {}

    for obj in bpy.context.scene.objects:
        if kl in obj:
            out[obj.name] = (tuple(obj[kl]), tuple(obj.get(kr, (0.0, 0.0, 0.0))))

    return out


def _read_current_world() -> dict:
    """
    Legge la posa in spazio mondo corrente di tutti gli oggetti MESH
    dalla scena Blender aperta.

    Returns:
        Dizionario {nome_oggetto: (loc_tuple, rot_tuple)}.
    """
    import bpy  # noqa: PLC0415

    out = {}
    for obj in bpy.context.scene.objects:
        if obj.type in {"CAMERA", "LIGHT"}:
            continue
        m = obj.matrix_world
        out[obj.name] = (tuple(m.translation), tuple(m.to_euler("XYZ")))

    return out


def _yaw_delta_deg(rot_a: tuple, rot_b: tuple) -> float:
    """
    Calcola la differenza di rotazione attorno all'asse Z tra due rotazioni
    Euler, normalizzata nell'intervallo [0, 180] gradi.

    Funzione pura.

    Args:
        rot_a: Tupla (rx, ry, rz) della rotazione A in radianti.
        rot_b: Tupla (rx, ry, rz) della rotazione B in radianti.

    Returns:
        Differenza assoluta di yaw in gradi, nell'intervallo [0, 180].
    """
    import math  # noqa: PLC0415
    d = math.degrees(rot_b[2] - rot_a[2])
    return abs((d + 180.0) % 360.0 - 180.0)


def _dist_xy(loc_a: tuple, loc_b: tuple) -> float:
    """
    Calcola la distanza euclidea orizzontale (piano XY) tra due posizioni.

    Funzione pura.

    Args:
        loc_a: Tupla (x, y, z) della posizione A.
        loc_b: Tupla (x, y, z) della posizione B.

    Returns:
        Distanza in metri nel piano XY.
    """
    dx = loc_b[0] - loc_a[0]
    dy = loc_b[1] - loc_a[1]
    return (dx * dx + dy * dy) ** 0.5


def _fmt(v: float, w: int, dec: int) -> str:
    """
    Formatta un numero float in una stringa a larghezza fissa.
    Restituisce un trattino destro-allineato se il valore e' NaN.

    Args:
        v:   Valore da formattare.
        w:   Larghezza totale della stringa.
        dec: Numero di cifre decimali.

    Returns:
        Stringa formattata.
    """
    return f"{'-':>{w}}" if v != v else f"{v:>{w}.{dec}f}"


def format_metrics_report(orig: dict, rand: dict, cur: dict) -> str:
    """
    Produce una tabella testuale con le metriche di spostamento degli oggetti,
    confrontando tre stati della scena:
        O = originale (prima del randomize),
        R = randomizzato,
        C = corrente (dopo la riorganizzazione dell'LLM).

    Le colonne mostrano:
        O->R d:   distanza di spostamento XY dal originale al randomizzato.
        R->C d:   distanza di spostamento XY dal randomizzato al corrente.
        R->C rot: variazione di rotazione attorno a Z (gradi).

    Funzione pura: riceve dizionari, non accede a bpy.

    Args:
        orig: Snapshot dello stato originale (da _read_snapshot).
        rand: Snapshot dello stato randomizzato (da _read_snapshot).
        cur:  Posa corrente degli oggetti (da _read_current_world).

    Returns:
        Stringa multiriga con il report formattato.
    """
    lines = [
        "NL2Scene3D - Metriche di spostamento (metri, gradi; piano XY)",
        "=" * 58,
    ]

    if not rand:
        lines.append("Nessuno snapshot 'randomizzato'. Esegui prima 'Randomize Layout',")
        lines.append("poi applica una risposta: le metriche compaiono qui.")
        return "\n".join(lines)

    names = sorted(set(rand) | set(orig) | set(cur))
    header = f"{'oggetto':28} {'O->R d':>8} {'R->C d':>8} {'R->C rot':>9}"
    lines += [header, "-" * len(header)]

    tot_rc = 0.0
    moved = 0

    for nm in names:
        o, r, c = orig.get(nm), rand.get(nm), cur.get(nm)
        d_or = _dist_xy(o[0], r[0]) if (o and r) else float("nan")
        d_rc = _dist_xy(r[0], c[0]) if (r and c) else float("nan")
        yaw  = _yaw_delta_deg(r[1], c[1]) if (r and c) else float("nan")
        lines.append(
            f"{nm[:28]:28} {_fmt(d_or, 8, 3)} {_fmt(d_rc, 8, 3)} {_fmt(yaw, 9, 1)}"
        )
        if (r and c) and d_rc == d_rc:  # Esclude NaN.
            tot_rc += d_rc
            if d_rc > 1e-4:
                moved += 1

    lines.append("-" * len(header))
    avg = (tot_rc / moved) if moved else 0.0
    lines.append(f"Oggetti spostati dall'LLM (R->C): {moved}")
    lines.append(f"Spostamento totale R->C: {tot_rc:.3f} m   medio: {avg:.3f} m")
    lines += [
        "",
        "Legenda: O=originale, R=randomizzato, C=corrente (riordino).",
        "O->R = quanto hai disordinato; R->C = quanto ha mosso l'LLM.",
        "d = distanza orizzontale; rot = variazione di rotazione attorno a Z.",
    ]

    return "\n".join(lines)


def build_metrics_report() -> str:
    """
    Legge gli snapshot originale (O) e randomizzato (R) dalle custom property
    e la posa corrente dalla scena Blender, quindi produce il report delle
    metriche di spostamento.

    Returns:
        Stringa con il report formattato prodotto da format_metrics_report.
    """
    return format_metrics_report(
        _read_snapshot("m_orig"),
        _read_snapshot("m_rand"),
        _read_current_world(),
    )
