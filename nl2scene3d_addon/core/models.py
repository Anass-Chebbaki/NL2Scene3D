# nl2scene3d/core/models.py
"""
Modelli dati condivisi di NL2Scene3D.

Principi di progetto:
    - Ogni dataclass e' autonoma e serializzabile in JSON.
    - Nessuna dipendenza da bpy o da altri moduli del package.
    - Gli helper geometrici di base (AABB, OBB, z_range) vivono accanto al dato
      a cui si riferiscono.

Questo modulo rappresenta il contratto dati di tutto l'add-on: SceneLoader,
randomizer, reorganizer e applicator producono e consumano queste classi.
I nomi dei campi e i metodi pubblici vanno mantenuti stabili.
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
    Trasformazione spaziale completa di un oggetto di scena.

    Attributi:
        location:       Posizione [x, y, z] dell'origine in coordinate mondo (m).
        rotation_euler: Rotazione [rx, ry, rz] in radianti, ordine XYZ.
        dimensions:     Dimensioni del bounding box [larghezza, profondita', altezza] (m).
        origin_offset:  Offset dell'origine rispetto al centro geometrico, in coordinate
                        locali gia' scalate per lo scale dell'oggetto (m).
    """

    location:       list[float]
    rotation_euler: list[float]
    dimensions:     list[float]
    origin_offset:  list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        for attr in ("location", "rotation_euler", "dimensions", "origin_offset"):
            if len(getattr(self, attr)) != 3:
                raise ValueError(f"'{attr}' deve contenere esattamente 3 valori.")

    # ------------------------------------------------------------------
    # Helper geometrici
    # ------------------------------------------------------------------

    def geometric_center_xy(self) -> tuple[float, float]:
        """
        Centro geometrico reale nel piano XY.

        Tiene conto dell'origin offset ruotato per la rotazione Z corrente,
        cosi' il centro e' sempre accurato indipendentemente dall'orientamento.
        """
        rz          = self.rotation_euler[2]
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        off          = self.origin_offset
        world_off_x  = off[0] * cos_z - off[1] * sin_z
        world_off_y  = off[0] * sin_z + off[1] * cos_z
        return self.location[0] + world_off_x, self.location[1] + world_off_y

    def aabb_xy(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        """
        Axis-Aligned Bounding Box nel piano XY.

        Restituisce (x_min, x_max, y_min, y_max) incluso il margine richiesto.
        Tiene conto della rotazione Z per calcolare l'estensione corretta.
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
        Quattro angoli dell'Oriented Bounding Box nel piano XY.

        Usato per il test di collisione con il Separating Axis Theorem (SAT).
        Il margine espande l'OBB su tutti i lati prima di restituire i vertici.
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
        """
        Estensione verticale dell'oggetto (z_min, z_max).

        Tiene conto dell'offset Z dell'origine rispetto al centro geometrico.
        """
        center_z = self.location[2] + self.origin_offset[2]
        half_h   = self.dimensions[2] / 2.0
        return center_z - half_h, center_z + half_h

    # ------------------------------------------------------------------
    # Serializzazione
    # ------------------------------------------------------------------

    def copy(self) -> "Transform":
        """Restituisce una copia profonda di questo Transform."""
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
    Un singolo oggetto nella scena 3D.

    Attributi:
        name:        Identificatore univoco (corrisponde al nome dell'oggetto in Blender).
        object_type: Tipo Blender dell'oggetto ('MESH', 'LIGHT', 'CAMERA', ...).
        transform:   Trasformazione spaziale corrente.
        category:    Categoria semantica ('structural', 'object', 'technical').
        is_movable:  Se True l'add-on puo' riposizionare questo oggetto.
        parent:      Nome dell'oggetto padre del gruppo, oppure None se e' un root.
        children:    Nomi dei figli diretti (popolato solo per gli oggetti root).
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
        """True se l'oggetto non ha un padre."""
        return self.parent is None

    @property
    def is_structural(self) -> bool:
        """True se l'oggetto e' classificato come strutturale."""
        return self.category == "structural"

    def copy(self) -> "SceneObject":
        """Restituisce una copia profonda di questo SceneObject."""
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
    Confini spaziali della stanza.

    Attributi:
        x_min, x_max: Intervallo sull'asse X (m).
        y_min, y_max: Intervallo sull'asse Y (m).
        z_floor:      Quota del pavimento (default 0.0 m).
        z_ceiling:    Quota del soffitto (default 3.0 m).
    """

    x_min:     float
    x_max:     float
    y_min:     float
    y_max:     float
    z_floor:   float = 0.0
    z_ceiling: float = 3.0

    @property
    def width(self) -> float:
        """Ampiezza della stanza sull'asse X."""
        return self.x_max - self.x_min

    @property
    def depth(self) -> float:
        """Profondita' della stanza sull'asse Y."""
        return self.y_max - self.y_min

    @property
    def height(self) -> float:
        """Altezza della stanza (soffitto - pavimento)."""
        return self.z_ceiling - self.z_floor

    @property
    def center_xy(self) -> tuple[float, float]:
        """Centro della stanza nel piano XY."""
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    def clamp_location(
        self,
        location:   list[float],
        dimensions: Optional[list[float]] = None,
        margin:     float = 0.0,
    ) -> list[float]:
        """
        Vincola X e Y per tenere l'oggetto completamente dentro i confini della stanza.

        Tiene conto delle dimensioni dell'oggetto e di un margine aggiuntivo.
        La coordinata Z non viene mai modificata.
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
        """True se l'AABB (x_min, x_max, y_min, y_max) e' completamente dentro i confini."""
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
            x_min=data["x_min"],     x_max=data["x_max"],
            y_min=data["y_min"],     y_max=data["y_max"],
            z_floor=data.get("z_floor",   0.0),
            z_ceiling=data.get("z_ceiling", 3.0),
        )


# ---------------------------------------------------------------------------
# SceneState
# ---------------------------------------------------------------------------

@dataclass
class SceneState:
    """
    Snapshot completo della scena a un determinato passo della pipeline.

    Attributi:
        scene_name:    Nome identificativo della scena.
        objects:       Lista di tutti gli oggetti (strutturali + mobili).
        room_bounds:   Confini spaziali calcolati per la stanza.
        pipeline_step: Etichetta del passo corrente ('original', 'randomized', 'reorganized').
        metadata:      Dizionario con dati extra (contatori, configurazioni, ...).
    """

    scene_name:    str
    objects:       list[SceneObject]
    room_bounds:   Optional[RoomBounds] = None
    pipeline_step: str                  = "unknown"
    metadata:      dict                 = field(default_factory=dict)

    # Cache interna nome -> oggetto (non serializzata, ricostruita automaticamente).
    _cache: dict[str, SceneObject] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        """Ricostruisce la cache nome -> SceneObject."""
        self._cache = {obj.name: obj for obj in self.objects}

    def get(self, name: str) -> Optional[SceneObject]:
        """Restituisce l'oggetto con il nome dato, o None se non trovato."""
        return self._cache.get(name)

    @property
    def movable_objects(self) -> list[SceneObject]:
        """Lista degli oggetti mobili."""
        return [o for o in self.objects if o.is_movable]

    @property
    def static_objects(self) -> list[SceneObject]:
        """Lista degli oggetti fissi."""
        return [o for o in self.objects if not o.is_movable]

    @property
    def root_movable_objects(self) -> list[SceneObject]:
        """Oggetti mobili senza padre (radici dei gruppi)."""
        return [o for o in self.objects if o.is_movable and o.is_root]

    def copy(self) -> "SceneState":
        """Restituisce una copia profonda di questo SceneState."""
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