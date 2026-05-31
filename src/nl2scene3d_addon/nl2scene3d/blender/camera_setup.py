# src/nl2scene3d/blender/camera_setup.py
"""
Camera configuration and placement for Blender rendering.

Manages the creation of standard views:
  - Top-down:        orthographic view from above for 2D layout verification.
  - Isometric:       perspective view from a configurable angle for preview rendering.
  - Isometric alt:   second angle for multi-view visual critic.
  - Front-low:       low-elevation view to validate object heights and Z relations.

After the first positioning call, the camera is "frozen" to guarantee
consistent framing across all pipeline steps.

Must be executed inside Blender's embedded Python environment.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import bpy
    from bpy.types import Object, Scene
    from nl2scene3d.config import RenderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen camera state
# ---------------------------------------------------------------------------

@dataclass
class FrozenCameraState:
    """
    Stores the frozen camera position to ensure consistent renders.

    Once the camera position is computed for a scene, the same position
    is reused for all subsequent pipeline steps.
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
        """Freeze the state so that subsequent placements reuse these values."""
        self.is_frozen = True
        logger.info("Camera state frozen. All subsequent renders will use the same framing.")


# Global singleton for the frozen camera state.
_frozen_state = FrozenCameraState()


def get_frozen_state() -> FrozenCameraState:
    """Return the global frozen camera state."""
    return _frozen_state


def reset_frozen_state() -> None:
    """Reset the frozen camera state (call this when starting a new scene)."""
    global _frozen_state
    _frozen_state = FrozenCameraState()
    logger.debug("Camera state reset.")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _get_scene_center_and_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> tuple[tuple[float, float, float], float]:
    """
    Compute the scene center and its largest dimension.

    Args:
        x_min: Minimum X bound.
        x_max: Maximum X bound.
        y_min: Minimum Y bound.
        y_max: Maximum Y bound.
        z_min: Minimum Z bound.
        z_max: Maximum Z bound.

    Returns:
        Tuple ((cx, cy, cz), max_dimension).
    """
    center = (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        (z_min + z_max) / 2.0,
    )
    max_dimension = max(x_max - x_min, y_max - y_min, z_max - z_min)
    return center, max_dimension


# ---------------------------------------------------------------------------
# Top-down camera
# ---------------------------------------------------------------------------

def setup_topdown_camera(
    scene_x_min: float,
    scene_x_max: float,
    scene_y_min: float,
    scene_y_max: float,
    scene_z_ceiling: float,
    config: "RenderConfig",
) -> None:
    """
    Position the active camera in orthographic top-down mode.

    The camera is placed above the room center, oriented downward, with
    orthographic scale sized to cover the entire room footprint.

    Args:
        scene_x_min:    Minimum X bound of the room.
        scene_x_max:    Maximum X bound of the room.
        scene_y_min:    Minimum Y bound of the room.
        scene_y_max:    Maximum Y bound of the room.
        scene_z_ceiling: Ceiling height (maximum Z).
        config:         Render configuration.

    Raises:
        ImportError: If bpy or mathutils are not available.
    """
    try:
        import bpy       # noqa: PLC0415
        import mathutils # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Modules 'bpy' and 'mathutils' require the Blender environment."
        ) from exc

    scene = bpy.context.scene
    state = get_frozen_state()

    camera_obj = _get_or_create_pipeline_camera(
        scene, "topdown", config.pipeline_camera_prefix
    )

    if state.is_frozen and state.topdown_location is not None:
        # Reuse the previously frozen position.
        camera_obj.location = mathutils.Vector(state.topdown_location)
        camera_obj.rotation_euler = mathutils.Euler(state.topdown_rotation)
        camera_data = camera_obj.data
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = state.topdown_ortho_scale
        scene.camera = camera_obj
        logger.debug("Top-down camera reused from frozen state.")
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

    # Persist to the freezable state.
    state.topdown_location = tuple(camera_obj.location)
    state.topdown_rotation = tuple(camera_obj.rotation_euler)
    state.topdown_ortho_scale = camera_data.ortho_scale

    logger.info(
        "Top-down camera configured: location=(%.2f, %.2f, %.2f), ortho_scale=%.2f.",
        center_x,
        center_y,
        camera_z,
        camera_data.ortho_scale,
    )


# ---------------------------------------------------------------------------
# Isometric camera (shared implementation)
# ---------------------------------------------------------------------------

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
    Internal implementation for positioning a perspective isometric camera.

    Args:
        scene_x_min:       Minimum X bound of the room.
        scene_x_max:       Maximum X bound of the room.
        scene_y_min:       Minimum Y bound of the room.
        scene_y_max:       Maximum Y bound of the room.
        scene_z_min:       Floor height (minimum Z).
        scene_z_ceiling:   Ceiling height.
        config:            Render configuration.
        azimuth_override:  Custom azimuth angle in degrees. Uses config value if None.
        elevation_override: Custom elevation angle in degrees. Uses config value if None.
        camera_suffix:     Suffix used to name the camera object.

    Raises:
        ImportError: If bpy or mathutils are not available.
    """
    try:
        import bpy       # noqa: PLC0415
        import mathutils # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Modules 'bpy' and 'mathutils' require the Blender environment."
        ) from exc

    scene = bpy.context.scene
    state = get_frozen_state()

    # Select which frozen-state slot to use based on the camera suffix.
    if camera_suffix == "isometric":
        frozen_loc = state.iso_location
        frozen_rot = state.iso_rotation
        frozen_fl  = state.iso_focal_length
    elif camera_suffix == "isometric2":
        frozen_loc = state.iso2_location
        frozen_rot = state.iso2_rotation
        frozen_fl  = state.iso2_focal_length
    elif camera_suffix == "front":
        frozen_loc = state.front_location
        frozen_rot = state.front_rotation
        frozen_fl  = state.front_focal_length
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
        logger.debug("Camera '%s' reused from frozen state.", camera_suffix)
        return

    center, _ = _get_scene_center_and_bounds(
        scene_x_min, scene_x_max,
        scene_y_min, scene_y_max,
        scene_z_min, scene_z_ceiling,
    )

    # Use horizontal extent (width or depth) rather than Z height for distance.
    horizontal_max = max(scene_x_max - scene_x_min, scene_y_max - scene_y_min)
    camera_distance = horizontal_max * config.isometric_distance_multiplier

    elevation = elevation_override if elevation_override is not None else config.isometric_elevation
    azimuth   = azimuth_override   if azimuth_override   is not None else config.isometric_azimuth

    elevation_rad = math.radians(elevation)
    azimuth_rad   = math.radians(azimuth)

    camera_x = center[0] + camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    camera_y = center[1] + camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    camera_z = center[2] + camera_distance * math.sin(elevation_rad)

    camera_obj.location = mathutils.Vector((camera_x, camera_y, camera_z))

    direction = (
        mathutils.Vector(center)
        - mathutils.Vector((camera_x, camera_y, camera_z))
    )
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera_obj.rotation_euler = rot_quat.to_euler()

    camera_data = camera_obj.data
    camera_data.type = "PERSP"
    camera_data.lens = config.isometric_focal_length

    scene.camera = camera_obj

    # Persist to the freezable state.
    loc_tuple = tuple(camera_obj.location)
    rot_tuple = tuple(camera_obj.rotation_euler)
    fl_value  = camera_data.lens

    if camera_suffix == "isometric":
        state.iso_location    = loc_tuple
        state.iso_rotation    = rot_tuple
        state.iso_focal_length = fl_value
    elif camera_suffix == "isometric2":
        state.iso2_location    = loc_tuple
        state.iso2_rotation    = rot_tuple
        state.iso2_focal_length = fl_value
    elif camera_suffix == "front":
        state.front_location    = loc_tuple
        state.front_rotation    = rot_tuple
        state.front_focal_length = fl_value

    logger.info(
        "Camera '%s' configured: location=(%.2f, %.2f, %.2f), azimuth=%.1f deg, elevation=%.1f deg.",
        camera_suffix,
        camera_x,
        camera_y,
        camera_z,
        azimuth,
        elevation,
    )


# ---------------------------------------------------------------------------
# Public camera setup functions
# ---------------------------------------------------------------------------

def setup_isometric_camera(
    scene_x_min: float,
    scene_x_max: float,
    scene_y_min: float,
    scene_y_max: float,
    scene_z_min: float,
    scene_z_ceiling: float,
    config: "RenderConfig",
) -> None:
    """Position the active camera in perspective isometric mode (primary angle)."""
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
    Position the active camera in perspective isometric mode (opposite angle).

    The azimuth is rotated ~160 degrees relative to the primary angle
    to show the opposite side of the room.
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
    Position a low-elevation front camera for validating object heights.

    Uses a 15-degree elevation angle to produce a near-horizontal view.
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


# ---------------------------------------------------------------------------
# Camera object management
# ---------------------------------------------------------------------------

def _get_or_create_pipeline_camera(
    scene: "Scene",
    suffix: str,
    prefix: str,
) -> "Object":
    """
    Retrieve or create a dedicated pipeline camera object.

    Args:
        scene:  Current Blender scene.
        suffix: Suffix that distinguishes cameras (e.g. 'topdown', 'isometric').
        prefix: Prefix for all pipeline camera names.

    Returns:
        The Blender camera object.
    """
    import bpy  # noqa: PLC0415

    camera_name = f"{prefix}_{suffix}"
    camera_obj = scene.objects.get(camera_name)

    if camera_obj is not None:
        logger.debug("Existing camera retrieved from scene: %s.", camera_name)
    else:
        if camera_name in bpy.data.objects:
            # Camera exists globally but is not linked to the current scene.
            camera_obj = bpy.data.objects[camera_name]
            scene.collection.objects.link(camera_obj)
            logger.debug("Global camera linked to scene: %s.", camera_name)
        else:
            camera_data = bpy.data.cameras.new(name=camera_name)
            camera_obj = bpy.data.objects.new(camera_name, camera_data)
            scene.collection.objects.link(camera_obj)
            logger.debug("New camera created: %s.", camera_name)

    return camera_obj