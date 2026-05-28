# src/nl2scene3d/blender/camera_setup.py
"""
Configurazione e posizionamento delle camere di rendering in Blender.

Gestisce la creazione di viste standard:
- Vista top-down: orthografica dall'alto per la verifica del layout 2D
- Vista isometrica: prospettica da angolo configurabile per anteprima fotorealistica
- Vista isometrica alternativa: secondo angolo per la visual critic multi-view
- Vista frontale bassa: per validare altezze e relazioni Z

La camera viene "congelata" dopo il primo posizionamento per garantire
consistenza tra tutti gli step della pipeline.

Deve essere eseguito all'interno dell'ambiente Python di Blender.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import bpy
    from bpy.types import Object, Scene
    from nl2scene3d.config import RenderConfig

logger = logging.getLogger(__name__)


@dataclass
class FrozenCameraState:
    """
    Stato congelato della camera per garantire consistenza tra render.
    
    Una volta calcolata la posizione della camera per la scena,
    la stessa posizione viene riutilizzata per tutti gli step successivi.
    """
    topdown_location: Optional[tuple[float, float, float]] = None
    topdown_rotation: Optional[tuple[float, float, float]] = None
    topdown_ortho_scale: Optional[float] = None
    
    iso_location: Optional[tuple[float, float, float]] = None
    iso_rotation: Optional[tuple[float, float, float]] = None
    iso_focal_length: Optional[float] = None
    
    iso2_location: Optional[tuple[float, float, float]] = None
    iso2_rotation: Optional[tuple[float, float, float]] = None
    iso2_focal_length: Optional[float] = None
    
    front_location: Optional[tuple[float, float, float]] = None
    front_rotation: Optional[tuple[float, float, float]] = None
    front_focal_length: Optional[float] = None
    
    is_frozen: bool = False

    def freeze(self) -> None:
        """Congela lo stato: i posizionamenti successivi riuseranno questi valori."""
        self.is_frozen = True
        logger.info("Camera state congelato. Tutti i render successivi useranno la stessa inquadratura.")


# Singleton globale per la camera congelata
_frozen_state = FrozenCameraState()


def get_frozen_state() -> FrozenCameraState:
    """Restituisce lo stato congelato della camera."""
    return _frozen_state


def reset_frozen_state() -> None:
    """Resetta lo stato congelato (per nuove scene)."""
    global _frozen_state
    _frozen_state = FrozenCameraState()
    logger.debug("Camera state resettato.")


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
    state = get_frozen_state()
    
    camera_obj = _get_or_create_pipeline_camera(
        scene, "topdown", config.pipeline_camera_prefix
    )
    
    if state.is_frozen and state.topdown_location is not None:
        # Riusa la posizione congelata
        camera_obj.location = mathutils.Vector(state.topdown_location)
        camera_obj.rotation_euler = mathutils.Euler(state.topdown_rotation)
        camera_data = camera_obj.data
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = state.topdown_ortho_scale
        scene.camera = camera_obj
        logger.debug("Camera top-down riusata da stato congelato.")
        return
    
    center_x = (scene_x_min + scene_x_max) / 2.0
    center_y = (scene_y_min + scene_y_max) / 2.0
    camera_z = scene_z_ceiling * config.topdown_height_multiplier

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
    
    # Salva nello stato congelabile
    state.topdown_location = tuple(camera_obj.location)
    state.topdown_rotation = tuple(camera_obj.rotation_euler)
    state.topdown_ortho_scale = camera_data.ortho_scale

    logger.info(
        "Camera top-down configurata: posizione=(%.2f, %.2f, %.2f), ortho_scale=%.2f.",
        center_x,
        center_y,
        camera_z,
        camera_data.ortho_scale,
    )


def _setup_isometric_camera_impl(
    scene_x_min: float,
    scene_x_max: float,
    scene_y_min: float,
    scene_y_max: float,
    scene_z_min: float,
    scene_z_ceiling: float,
    config: "RenderConfig",
    azimuth_override: Optional[float] = None,
    elevation_override: Optional[float] = None,
    camera_suffix: str = "isometric",
) -> None:
    """
    Implementazione interna per posizionare una camera isometrica prospettica.

    Args:
        scene_x_min: Limite minimo sull'asse X della stanza.
        scene_x_max: Limite massimo sull'asse X della stanza.
        scene_y_min: Limite minimo sull'asse Y della stanza.
        scene_y_max: Limite massimo sull'asse Y della stanza.
        scene_z_min: Quota minima (pavimento).
        scene_z_ceiling: Altezza del soffitto.
        config: Configurazione del rendering.
        azimuth_override: Angolo azimutale custom (gradi). Se None, usa config.
        elevation_override: Angolo elevazione custom (gradi). Se None, usa config.
        camera_suffix: Suffisso per il nome della camera.

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
    state = get_frozen_state()
    
    # Determina quale slot dello stato congelato usare
    if camera_suffix == "isometric":
        frozen_loc = state.iso_location
        frozen_rot = state.iso_rotation
        frozen_fl = state.iso_focal_length
    elif camera_suffix == "isometric2":
        frozen_loc = state.iso2_location
        frozen_rot = state.iso2_rotation
        frozen_fl = state.iso2_focal_length
    elif camera_suffix == "front":
        frozen_loc = state.front_location
        frozen_rot = state.front_rotation
        frozen_fl = state.front_focal_length
    else:
        frozen_loc = frozen_rot = frozen_fl = None

    camera_obj = _get_or_create_pipeline_camera(
        scene, camera_suffix, config.pipeline_camera_prefix
    )
    
    if state.is_frozen and frozen_loc is not None:
        camera_obj.location = mathutils.Vector(frozen_loc)
        camera_obj.rotation_euler = mathutils.Euler(frozen_rot)
        camera_data = camera_obj.data
        camera_data.type = "PERSP"
        camera_data.lens = frozen_fl
        scene.camera = camera_obj
        logger.debug("Camera '%s' riusata da stato congelato.", camera_suffix)
        return

    center, _ = _get_scene_center_and_bounds(
        scene_x_min, scene_x_max,
        scene_y_min, scene_y_max,
        scene_z_min, scene_z_ceiling,
    )

    # Usa la dimensione orizzontale (max tra width e depth) - non l'altezza Z.
    horizontal_max = max(scene_x_max - scene_x_min, scene_y_max - scene_y_min)
    camera_distance = horizontal_max * config.isometric_distance_multiplier
    
    elevation = elevation_override if elevation_override is not None else config.isometric_elevation
    azimuth = azimuth_override if azimuth_override is not None else config.isometric_azimuth
    
    elevation_rad = math.radians(elevation)
    azimuth_rad = math.radians(azimuth)

    camera_x = center[0] + camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    camera_y = center[1] + camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    camera_z = center[2] + camera_distance * math.sin(elevation_rad)

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
    
    # Salva nello stato congelabile
    loc_tuple = tuple(camera_obj.location)
    rot_tuple = tuple(camera_obj.rotation_euler)
    fl_value = camera_data.lens
    
    if camera_suffix == "isometric":
        state.iso_location = loc_tuple
        state.iso_rotation = rot_tuple
        state.iso_focal_length = fl_value
    elif camera_suffix == "isometric2":
        state.iso2_location = loc_tuple
        state.iso2_rotation = rot_tuple
        state.iso2_focal_length = fl_value
    elif camera_suffix == "front":
        state.front_location = loc_tuple
        state.front_rotation = rot_tuple
        state.front_focal_length = fl_value

    logger.info(
        "Camera '%s' configurata: posizione=(%.2f, %.2f, %.2f), azimuth=%.1f°, elevation=%.1f°.",
        camera_suffix,
        camera_x,
        camera_y,
        camera_z,
        azimuth,
        elevation,
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
    Posiziona la camera attiva in modalita' isometrica prospettica (angolo primario).
    """
    _setup_isometric_camera_impl(
        scene_x_min, scene_x_max,
        scene_y_min, scene_y_max,
        scene_z_min, scene_z_ceiling,
        config,
        camera_suffix="isometric",
    )


def setup_isometric_camera_angle2(
    scene_x_min: float,
    scene_x_max: float,
    scene_y_min: float,
    scene_y_max: float,
    scene_z_min: float,
    scene_z_ceiling: float,
    config: "RenderConfig",
) -> None:
    """
    Posiziona la camera attiva in modalita' isometrica prospettica (angolo opposto).
    L'angolo e' ruotato di ~180° rispetto al primario per mostrare il lato opposto della stanza.
    """
    opposite_azimuth = (config.isometric_azimuth + 160.0) % 360.0
    _setup_isometric_camera_impl(
        scene_x_min, scene_x_max,
        scene_y_min, scene_y_max,
        scene_z_min, scene_z_ceiling,
        config,
        azimuth_override=opposite_azimuth,
        elevation_override=config.isometric_elevation,
        camera_suffix="isometric2",
    )


def setup_front_camera(
    scene_x_min: float,
    scene_x_max: float,
    scene_y_min: float,
    scene_y_max: float,
    scene_z_min: float,
    scene_z_ceiling: float,
    config: "RenderConfig",
) -> None:
    """
    Posiziona una camera frontale bassa per validare le altezze degli oggetti.
    Angolo di elevazione basso (15°) per una vista quasi orizzontale.
    """
    _setup_isometric_camera_impl(
        scene_x_min, scene_x_max,
        scene_y_min, scene_y_max,
        scene_z_min, scene_z_ceiling,
        config,
        azimuth_override=(config.isometric_azimuth + 80.0) % 360.0,
        elevation_override=15.0,
        camera_suffix="front",
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

    camera_obj = scene.objects.get(camera_name)
    if camera_obj is not None:
        logger.debug("Camera esistente recuperata dalla scena: %s.", camera_name)
    else:
        # Se esiste nei dati globali ma non nella scena corrente, colleghiamola
        if camera_name in bpy.data.objects:
            camera_obj = bpy.data.objects[camera_name]
            scene.collection.objects.link(camera_obj)
            logger.debug("Camera globale collegata alla scena: %s.", camera_name)
        else:
            camera_data = bpy.data.cameras.new(name=camera_name)
            camera_obj = bpy.data.objects.new(camera_name, camera_data)
            scene.collection.objects.link(camera_obj)
            logger.debug("Nuova camera creata: %s.", camera_name)

    return camera_obj