"""
Configurazione e posizionamento delle camere di rendering in Blender.

Gestisce la creazione di due viste standard per ogni step della pipeline:
- Vista top-down: orthografica dall'alto per la verifica del layout 2D
- Vista isometrica: prospettica a 45 gradi per l'anteprima fotorealistica

Deve essere eseguito all'interno dell'ambiente Python di Blender.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Angolo di elevazione della camera isometrica rispetto al piano XY
ISOMETRIC_ELEVATION_DEGREES: float = 45.0

# Angolo azimutale della camera isometrica (vista da angolo frontale-laterale)
ISOMETRIC_AZIMUTH_DEGREES: float = 45.0

# Moltiplicatore dell'altezza della stanza per il posizionamento della camera top-down
TOP_DOWN_HEIGHT_MULTIPLIER: float = 3.0

# Nome prefisso per le camere create dalla pipeline (evita conflitti con camere esistenti)
PIPELINE_CAMERA_PREFIX: str = "NL2Scene3D_Camera"

def _get_scene_center_and_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> tuple[tuple[float, float, float], float]:
    """
    Calcola il centro e la dimensione massima della scena.

    Args:
        x_min, x_max: Limiti sull'asse X.
        y_min, y_max: Limiti sull'asse Y.
        z_min, z_max: Limiti sull'asse Z.

    Returns:
        Tupla (centro_xyz, dimensione_massima).
    """
    center = (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        (z_min + z_max) / 2.0,
    )
    max_dimension = max(x_max - x_min, y_max - y_min, z_max - z_min)
    return center, max_dimension

def setup_topdown_camera(
    scene_x_min: float,
    scene_x_max: float,
    scene_y_min: float,
    scene_y_max: float,
    scene_z_ceiling: float,
) -> None:
    """
    Posiziona la camera attiva in modalita' top-down orthografica.

    La camera viene posizionata direttamente sopra il centro della scena,
    orientata verso il basso con proiezione ortografica.

    Args:
        scene_x_min: Limite minimo sull'asse X della stanza.
        scene_x_max: Limite massimo sull'asse X della stanza.
        scene_y_min: Limite minimo sull'asse Y della stanza.
        scene_y_max: Limite massimo sull'asse Y della stanza.
        scene_z_ceiling: Altezza del soffitto (quota massima Z).

    Raises:
        ImportError: Se bpy non e' disponibile.
        RuntimeError: Se non e' presente nessuna camera nella scena.
    """
    try:
        import bpy  # noqa: PLC0415
        import mathutils  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "I moduli 'bpy' e 'mathutils' richiedono l'ambiente Blender."
        ) from exc

    scene = bpy.context.scene
    center_x = (scene_x_min + scene_x_max) / 2.0
    center_y = (scene_y_min + scene_y_max) / 2.0
    camera_z = scene_z_ceiling * TOP_DOWN_HEIGHT_MULTIPLIER

    # Cerca la camera pipeline o usa la camera attiva
    camera_obj = _get_or_create_pipeline_camera(scene, "topdown")

    # Posiziona la camera sopra il centro della stanza
    camera_obj.location = mathutils.Vector((center_x, center_y, camera_z))

    # Orienta la camera verso il basso: rotazione X di 90 gradi
    camera_obj.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), "XYZ")
    camera_obj.rotation_euler.x = math.radians(0.0)
    camera_obj.rotation_euler.y = math.radians(0.0)
    camera_obj.rotation_euler.z = math.radians(0.0)

    # Usa la trasformazione di "guarda verso il basso" tramite rotation
    camera_obj.rotation_euler.x = math.radians(0.0)  # punta verso -Z
    # In Blender la camera punta verso -Z di default, quindi per top-down:
    # rotation X = 0, Y = 0, Z = 0 con camera a Z elevata guarda verso -Z
    # ma in Blender la camera guarda verso -Z locale, quindi non serve rotazione aggiuntiva
    # per top-down standard. La rotazione corretta e' X=0, Y=0, Z=0 con camera alta.

    # Imposta la proiezione ortografica
    camera_data = camera_obj.data
    camera_data.type = "ORTHO"
    room_width = scene_x_max - scene_x_min
    room_depth = scene_y_max - scene_y_min
    # La scala ortografica deve coprire la stanza intera
    camera_data.ortho_scale = max(room_width, room_depth) * 1.1

    scene.camera = camera_obj
    logger.info(
        "Camera top-down configurata: posizione=(%.2f, %.2f, %.2f), ortho_scale=%.2f",
        center_x,
        center_y,
        camera_z,
        camera_data.ortho_scale,
    )

def setup_isometric_camera(
    scene_x_min: float,
    scene_x_max: float,
    scene_y_min: float,
    scene_y_max: float,
    scene_z_min: float,
    scene_z_ceiling: float,
) -> None:
    """
    Posiziona la camera attiva in modalita' isometrica prospettica.

    La camera viene posizionata a 45 gradi di elevazione e azimuth,
    orientata verso il centro della scena.

    Args:
        scene_x_min: Limite minimo sull'asse X della stanza.
        scene_x_max: Limite massimo sull'asse X della stanza.
        scene_y_min: Limite minimo sull'asse Y della stanza.
        scene_y_max: Limite massimo sull'asse Y della stanza.
        scene_z_min: Quota minima (pavimento).
        scene_z_ceiling: Altezza del soffitto.

    Raises:
        ImportError: Se bpy non e' disponibile.
    """
    try:
        import bpy  # noqa: PLC0415
        import mathutils  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "I moduli 'bpy' e 'mathutils' richiedono l'ambiente Blender."
        ) from exc

    scene = bpy.context.scene
    center, max_dim = _get_scene_center_and_bounds(
        scene_x_min, scene_x_max,
        scene_y_min, scene_y_max,
        scene_z_min, scene_z_ceiling,
    )

    # Distanza della camera dal centro basata sulla dimensione massima della scena
    camera_distance = max_dim * 1.8

    # Calcola la posizione della camera in coordinate sferiche
    elevation_rad = math.radians(ISOMETRIC_ELEVATION_DEGREES)
    azimuth_rad = math.radians(ISOMETRIC_AZIMUTH_DEGREES)

    camera_x = center[0] + camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    camera_y = center[1] + camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    camera_z = center[2] + camera_distance * math.sin(elevation_rad)

    camera_obj = _get_or_create_pipeline_camera(scene, "isometric")
    camera_obj.location = mathutils.Vector((camera_x, camera_y, camera_z))

    # Orienta la camera verso il centro della scena
    direction = mathutils.Vector(center) - mathutils.Vector((camera_x, camera_y, camera_z))
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera_obj.rotation_euler = rot_quat.to_euler()

    # Imposta la proiezione prospettica
    camera_data = camera_obj.data
    camera_data.type = "PERSP"
    camera_data.lens = 35.0  # mm, equivalente a un grandangolo moderato

    scene.camera = camera_obj
    logger.info(
        "Camera isometrica configurata: posizione=(%.2f, %.2f, %.2f)",
        camera_x,
        camera_y,
        camera_z,
    )

def _get_or_create_pipeline_camera(
    scene: object,
    suffix: str,
) -> object:
    """
    Recupera o crea una camera dedicata alla pipeline.

    Usa una camera separata per non alterare le camere originali della scena.

    Args:
        scene: Scena Blender corrente (bpy.context.scene).
        suffix: Suffisso per distinguere le camere (es. 'topdown', 'isometric').

    Returns:
        Oggetto camera Blender.
    """
    import bpy  # noqa: PLC0415

    camera_name = f"{PIPELINE_CAMERA_PREFIX}_{suffix}"

    if camera_name in bpy.data.objects:
        camera_obj = bpy.data.objects[camera_name]
        logger.debug("Camera esistente recuperata: %s", camera_name)
    else:
        camera_data = bpy.data.cameras.new(name=camera_name)
        camera_obj = bpy.data.objects.new(camera_name, camera_data)
        scene.collection.objects.link(camera_obj)
        logger.debug("Nuova camera creata: %s", camera_name)

    return camera_obj
