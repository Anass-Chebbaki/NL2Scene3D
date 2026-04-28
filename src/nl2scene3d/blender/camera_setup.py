# src/nl2scene3d/blender/camera_setup.py
"""
Configurazione e posizionamento delle camere di rendering in Blender.

Gestisce la creazione di due viste standard:
- Vista top-down: orthografica dall'alto per la verifica del layout 2D
- Vista isometrica: prospettica a 45 gradi per l'anteprima fotorealistica

Deve essere eseguito all'interno dell'ambiente Python di Blender.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import bpy
    from bpy.types import Object, Scene
    from nl2scene3d.config import RenderConfig

logger = logging.getLogger(__name__)


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
        x_min: Limite minimo sull'asse X.
        x_max: Limite massimo sull'asse X.
        y_min: Limite minimo sull'asse Y.
        y_max: Limite massimo sull'asse Y.
        z_min: Limite minimo sull'asse Z.
        z_max: Limite massimo sull'asse Z.

    Returns:
        Tupla ((cx, cy, cz), dimensione_massima).
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
    config: "RenderConfig",
) -> None:
    """
    Posiziona la camera attiva in modalita' top-down orthografica.

    La camera viene posizionata sopra il centro della stanza, orientata
    verso il basso con proiezione ortografica scalata per coprire
    l'intera area della stanza.

    Args:
        scene_x_min: Limite minimo sull'asse X della stanza.
        scene_x_max: Limite massimo sull'asse X della stanza.
        scene_y_min: Limite minimo sull'asse Y della stanza.
        scene_y_max: Limite massimo sull'asse Y della stanza.
        scene_z_ceiling: Altezza del soffitto (quota massima Z).
        config: Configurazione del rendering.

    Raises:
        ImportError: Se bpy o mathutils non sono disponibili.
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
    camera_z = scene_z_ceiling * config.topdown_height_multiplier

    camera_obj = _get_or_create_pipeline_camera(
        scene, "topdown", config.pipeline_camera_prefix
    )

    camera_obj.location = mathutils.Vector((center_x, center_y, camera_z))

    direction = mathutils.Vector((center_x, center_y, 0.0)) - camera_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera_obj.rotation_euler = rot_quat.to_euler()

    camera_data = camera_obj.data
    camera_data.type = "ORTHO"
    room_width = scene_x_max - scene_x_min
    room_depth = scene_y_max - scene_y_min
    camera_data.ortho_scale = (
        max(room_width, room_depth) * config.topdown_ortho_scale_padding
    )

    scene.camera = camera_obj

    logger.info(
        "Camera top-down configurata: posizione=(%.2f, %.2f, %.2f), ortho_scale=%.2f.",
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
    config: "RenderConfig",
) -> None:
    """
    Posiziona la camera attiva in modalita' isometrica prospettica.

    La camera viene posizionata in base agli angoli di elevazione e azimuth
    definiti nella configurazione, orientata verso il centro della scena.

    Args:
        scene_x_min: Limite minimo sull'asse X della stanza.
        scene_x_max: Limite massimo sull'asse X della stanza.
        scene_y_min: Limite minimo sull'asse Y della stanza.
        scene_y_max: Limite massimo sull'asse Y della stanza.
        scene_z_min: Quota minima (pavimento).
        scene_z_ceiling: Altezza del soffitto.
        config: Configurazione del rendering.

    Raises:
        ImportError: Se bpy o mathutils non sono disponibili.
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

    # Usa la dimensione orizzontale (max tra width e depth) - non l'altezza Z.
    # Questo evita che stanze "alte" (es. con tetti enormi) facciano allontanare troppo la camera.
    horizontal_max = max(scene_x_max - scene_x_min, scene_y_max - scene_y_min)
    camera_distance = horizontal_max * config.isometric_distance_multiplier
    elevation_rad = math.radians(config.isometric_elevation)
    azimuth_rad = math.radians(config.isometric_azimuth)

    camera_x = center[0] + camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    camera_y = center[1] + camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    camera_z = center[2] + camera_distance * math.sin(elevation_rad)

    camera_obj = _get_or_create_pipeline_camera(
        scene, "isometric", config.pipeline_camera_prefix
    )

    camera_obj.location = mathutils.Vector((camera_x, camera_y, camera_z))

    direction = (
        mathutils.Vector(center) - mathutils.Vector((camera_x, camera_y, camera_z))
    )
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera_obj.rotation_euler = rot_quat.to_euler()

    camera_data = camera_obj.data
    camera_data.type = "PERSP"
    camera_data.lens = config.isometric_focal_length

    scene.camera = camera_obj

    logger.info(
        "Camera isometrica configurata: posizione=(%.2f, %.2f, %.2f).",
        camera_x,
        camera_y,
        camera_z,
    )


def _get_or_create_pipeline_camera(
    scene: "Scene",
    suffix: str,
    prefix: str,
) -> "Object":
    """
    Recupera o crea una camera dedicata alla pipeline.

    Args:
        scene: Scena Blender corrente.
        suffix: Suffisso per distinguere le camere ('topdown', 'isometric').
        prefix: Prefisso per i nomi delle camere.

    Returns:
        Oggetto camera Blender.
    """
    import bpy  # noqa: PLC0415

    camera_name = f"{prefix}_{suffix}"

    if camera_name in bpy.data.objects:
        camera_obj = bpy.data.objects[camera_name]
        logger.debug("Camera esistente recuperata: %s.", camera_name)
    else:
        camera_data = bpy.data.cameras.new(name=camera_name)
        camera_obj = bpy.data.objects.new(camera_name, camera_data)
        scene.collection.objects.link(camera_obj)
        logger.debug("Nuova camera creata: %s.", camera_name)

    return camera_obj