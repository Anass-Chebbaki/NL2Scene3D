# nl2scene3d/models.py
"""
Shared data models for the NL2Scene3D pipeline.

Design principles:
  - Every dataclass is self-contained and JSON-serializable.
  - No dependency on bpy or any other pipeline module.
  - Basic geometry helpers (AABB, OBB) live alongside their data.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

@dataclass
class Transform:
    """
    Full spatial transform of a scene object.

    Attributes:
        location:      Position [x, y, z] of the object origin in world coordinates.
        rotation_euler: Rotation [rx, ry, rz] in radians, XYZ order.
        dimensions:    Bounding-box dimensions [width, depth, height] in meters.
        origin_offset: Offset of the origin from the geometric center in local
                       coordinates, pre-scaled by the object scale (meters).
    """

    location:       list[float]
    rotation_euler: list[float]
    dimensions:     list[float]
    origin_offset:  list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        for attr in ("location", "rotation_euler", "dimensions", "origin_offset"):
            if len(getattr(self, attr)) != 3:
                raise ValueError(f"'{attr}' must contain exactly 3 values.")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def geometric_center_xy(self) -> tuple[float, float]:
        """
        Real geometric center in the XY plane, accounting for the origin
        offset rotated by the current Z rotation.
        """
        rz          = self.rotation_euler[2]
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        off         = self.origin_offset
        world_off_x = off[0] * cos_z - off[1] * sin_z
        world_off_y = off[0] * sin_z + off[1] * cos_z
        return self.location[0] + world_off_x, self.location[1] + world_off_y

    def aabb_xy(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        """
        Axis-Aligned Bounding Box in the XY plane.

        Returns:
            (x_min, x_max, y_min, y_max) including the requested margin.
        """
        cx, cy       = self.geometric_center_xy()
        rz           = self.rotation_euler[2]
        cos_z, sin_z = abs(math.cos(rz)), abs(math.sin(rz))
        dim          = self.dimensions

        eff_x  = dim[0] * cos_z + dim[1] * sin_z
        eff_y  = dim[0] * sin_z + dim[1] * cos_z
        half_x = eff_x / 2.0 + margin
        half_y = eff_y / 2.0 + margin

        return cx - half_x, cx + half_x, cy - half_y, cy + half_y

    def obb_corners_xy(self, margin: float = 0.0) -> list[tuple[float, float]]:
        """
        Four corners of the Oriented Bounding Box in the XY plane.

        Used for the Separating Axis Theorem (SAT) collision check.
        """
        cx, cy       = self.geometric_center_xy()
        rz           = self.rotation_euler[2]
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        dim          = self.dimensions
        w = dim[0] / 2.0 + margin
        h = dim[1] / 2.0 + margin
        local = [(-w, -h), (w, -h), (w, h), (-w, h)]
        return [
            (cx + lx * cos_z - ly * sin_z, cy + lx * sin_z + ly * cos_z)
            for lx, ly in local
        ]

    def z_range(self) -> tuple[float, float]:
        """Vertical extent (z_min, z_max), accounting for the Z origin offset."""
        center_z = self.location[2] + self.origin_offset[2]
        half_h   = self.dimensions[2] / 2.0
        return center_z - half_h, center_z + half_h

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def copy(self) -> "Transform":
        return Transform(
            location=list(self.location),
            rotation_euler=list(self.rotation_euler),
            dimensions=list(self.dimensions),
            origin_offset=list(self.origin_offset),
        )

    def to_dict(self) -> dict:
        return {
            "location":       self.location,
            "rotation_euler": self.rotation_euler,
            "dimensions":     self.dimensions,
            "origin_offset":  self.origin_offset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transform":
        return cls(
            location=list(data["location"]),
            rotation_euler=list(data["rotation_euler"]),
            dimensions=list(data["dimensions"]),
            origin_offset=list(data.get("origin_offset", [0.0, 0.0, 0.0])),
        )


# ---------------------------------------------------------------------------
# SceneObject
# ---------------------------------------------------------------------------

@dataclass
class SceneObject:
    """
    A single object in the 3D scene.

    Attributes:
        name:        Unique identifier (Blender object name).
        object_type: Blender type string ('MESH', 'LIGHT', 'CAMERA', ...).
        transform:   Current spatial transform.
        category:    Semantic category ('furniture', 'structural', 'decoration', ...).
        is_movable:  Whether the pipeline is allowed to reposition this object.
        parent:      Name of the parent group object, or None if this is a root.
        children:    Names of direct child objects (populated for root objects only).
    """

    name:        str
    object_type: str
    transform:   Transform
    category:    str           = "unknown"
    is_movable:  bool          = True
    parent:      Optional[str] = None
    children:    list[str]     = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        """True if the object has no parent."""
        return self.parent is None

    @property
    def is_structural(self) -> bool:
        return self.category == "structural"

    def copy(self) -> "SceneObject":
        return SceneObject(
            name=self.name,
            object_type=self.object_type,
            transform=self.transform.copy(),
            category=self.category,
            is_movable=self.is_movable,
            parent=self.parent,
            children=list(self.children),
        )

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "type":           self.object_type,
            "location":       self.transform.location,
            "rotation_euler": self.transform.rotation_euler,
            "dimensions":     self.transform.dimensions,
            "origin_offset":  self.transform.origin_offset,
            "category":       self.category,
            "is_movable":     self.is_movable,
            "parent":         self.parent,
            "children":       self.children,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneObject":
        return cls(
            name=data["name"],
            object_type=data["type"],
            transform=Transform.from_dict(data),
            category=data.get("category", "unknown"),
            is_movable=data.get("is_movable", True),
            parent=data.get("parent"),
            children=list(data.get("children", [])),
        )


# ---------------------------------------------------------------------------
# RoomBounds
# ---------------------------------------------------------------------------

@dataclass
class RoomBounds:
    """
    Spatial boundaries of the room.

    Attributes:
        x_min, x_max: Range on the X axis.
        y_min, y_max: Range on the Y axis.
        z_floor:      Floor elevation (default 0.0).
        z_ceiling:    Ceiling elevation.
    """

    x_min:     float
    x_max:     float
    y_min:     float
    y_max:     float
    z_floor:   float = 0.0
    z_ceiling: float = 3.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def depth(self) -> float:
        return self.y_max - self.y_min

    @property
    def height(self) -> float:
        return self.z_ceiling - self.z_floor

    @property
    def center_xy(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    def clamp_location(
        self,
        location:   list[float],
        dimensions: Optional[list[float]] = None,
        margin:     float = 0.0,
    ) -> list[float]:
        """
        Clamps X and Y to keep the object inside the room bounds.

        Takes object dimensions and an additional margin into account.
        The Z coordinate is never modified.
        """
        off_x = (dimensions[0] / 2.0 if dimensions else 0.0) + margin
        off_y = (dimensions[1] / 2.0 if dimensions else 0.0) + margin
        return [
            max(self.x_min + off_x, min(self.x_max - off_x, location[0])),
            max(self.y_min + off_y, min(self.y_max - off_y, location[1])),
            location[2],
        ]

    def contains_aabb(
        self,
        aabb:   tuple[float, float, float, float],
        margin: float = 0.0,
    ) -> bool:
        """True if the AABB (x_min, x_max, y_min, y_max) lies fully inside the bounds."""
        return (
            aabb[0] >= self.x_min + margin
            and aabb[1] <= self.x_max - margin
            and aabb[2] >= self.y_min + margin
            and aabb[3] <= self.y_max - margin
        )

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min, "x_max": self.x_max,
            "y_min": self.y_min, "y_max": self.y_max,
            "z_floor": self.z_floor, "z_ceiling": self.z_ceiling,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoomBounds":
        return cls(
            x_min=data["x_min"], x_max=data["x_max"],
            y_min=data["y_min"], y_max=data["y_max"],
            z_floor=data.get("z_floor", 0.0),
            z_ceiling=data.get("z_ceiling", 3.0),
        )


# ---------------------------------------------------------------------------
# SceneState
# ---------------------------------------------------------------------------

@dataclass
class SceneState:
    """
    Complete snapshot of the scene at a given pipeline stage.

    Attributes:
        scene_name:    Identifying name for the scene.
        objects:       All objects (structural + movable).
        room_bounds:   Computed spatial boundaries.
        pipeline_step: Step label ('original', 'randomized', 'reordered', 'refined').
        metadata:      Arbitrary extra data (counters, errors, ...).
    """

    scene_name:    str
    objects:       list[SceneObject]
    room_bounds:   Optional[RoomBounds] = None
    pipeline_step: str                  = "unknown"
    metadata:      dict                 = field(default_factory=dict)

    # Internal name-to-object cache — not serialized.
    _cache: dict[str, SceneObject] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        self._cache = {obj.name: obj for obj in self.objects}

    def get(self, name: str) -> Optional[SceneObject]:
        return self._cache.get(name)

    @property
    def movable_objects(self) -> list[SceneObject]:
        return [o for o in self.objects if o.is_movable]

    @property
    def static_objects(self) -> list[SceneObject]:
        return [o for o in self.objects if not o.is_movable]

    @property
    def root_movable_objects(self) -> list[SceneObject]:
        """Movable objects that have no parent (group roots)."""
        return [o for o in self.objects if o.is_movable and o.is_root]

    def copy(self) -> "SceneState":
        return SceneState(
            scene_name=self.scene_name,
            objects=[o.copy() for o in self.objects],
            room_bounds=copy.deepcopy(self.room_bounds),
            pipeline_step=self.pipeline_step,
            metadata=copy.deepcopy(self.metadata),
        )

    def to_dict(self) -> dict:
        return {
            "scene_name":    self.scene_name,
            "pipeline_step": self.pipeline_step,
            "room_bounds":   self.room_bounds.to_dict() if self.room_bounds else None,
            "objects":       [o.to_dict() for o in self.objects],
            "metadata":      self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneState":
        return cls(
            scene_name=data["scene_name"],
            objects=[SceneObject.from_dict(o) for o in data["objects"]],
            room_bounds=(
                RoomBounds.from_dict(data["room_bounds"])
                if data.get("room_bounds")
                else None
            ),
            pipeline_step=data.get("pipeline_step", "unknown"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# LLMCorrection
# ---------------------------------------------------------------------------

@dataclass
class LLMCorrection:
    """A position or rotation correction suggested by the vision LLM for one object."""

    object_name:       str
    action:            str                  # 'move' | 'rotate' | 'move_and_rotate'
    new_location:      Optional[list[float]] = None
    new_rotation_euler: Optional[list[float]] = None
    reason:            str                  = ""

    def to_dict(self) -> dict:
        return {
            "object_name":       self.object_name,
            "action":            self.action,
            "new_location":      self.new_location,
            "new_rotation_euler": self.new_rotation_euler,
            "reason":            self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMCorrection":
        return cls(
            object_name=data["object_name"],
            action=data["action"],
            new_location=data.get("new_location"),
            new_rotation_euler=data.get("new_rotation_euler"),
            reason=data.get("reason", ""),
        )