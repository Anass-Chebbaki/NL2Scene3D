"""
Definizione dei modelli dati condivisi in tutta la pipeline.

Vengono usati dataclass tipizzate per garantire coerenza strutturale
e la serializzazione JSON.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ObjectTransform:
    """
    Rappresenta la trasformazione completa di un oggetto nella scena.

    Attributes:
        location: Coordinate (x, y, z) nel sistema di riferimento globale.
        rotation_euler: Rotazione in radianti sugli assi (x, y, z), ordine XYZ.
        dimensions: Dimensioni del bounding box (larghezza, profondita', altezza).
    """

    location: list[float]
    rotation_euler: list[float]
    dimensions: list[float]

    def __post_init__(self) -> None:
        """Valida che le liste abbiano esattamente 3 componenti."""
        for attr_name in ("location", "rotation_euler", "dimensions"):
            value = getattr(self, attr_name)
            if len(value) != 3:
                raise ValueError(
                    f"'{attr_name}' deve contenere esattamente 3 valori, "
                    f"ricevuto: {len(value)}"
                )

    def to_dict(self) -> dict:
        """Serializza in dizionario compatibile con JSON."""
        return {
            "location": self.location,
            "rotation_euler": self.rotation_euler,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectTransform":
        """Deserializza da dizionario JSON."""
        return cls(
            location=list(data["location"]),
            rotation_euler=list(data["rotation_euler"]),
            dimensions=list(data["dimensions"]),
        )

@dataclass
class SceneObject:
    """
    Rappresenta un oggetto all'interno della scena 3D.

    Attributes:
        name: Identificatore univoco dell'oggetto nella scena Blender.
        object_type: Tipo Blender dell'oggetto (es. 'MESH', 'LIGHT', 'CAMERA').
        transform: Trasformazione corrente dell'oggetto.
        category: Categoria semantica (es. 'furniture', 'decoration', 'structural').
        is_movable: Se True, l'oggetto puo' essere spostato dalla pipeline.
    """

    name: str
    object_type: str
    transform: ObjectTransform
    category: str = "unknown"
    is_movable: bool = True

    def to_dict(self) -> dict:
        """Serializza in dizionario compatibile con JSON."""
        return {
            "name": self.name,
            "type": self.object_type,
            "location": self.transform.location,
            "rotation_euler": self.transform.rotation_euler,
            "dimensions": self.transform.dimensions,
            "category": self.category,
            "is_movable": self.is_movable,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneObject":
        """Deserializza da dizionario JSON."""
        transform = ObjectTransform(
            location=list(data["location"]),
            rotation_euler=list(data["rotation_euler"]),
            dimensions=list(data["dimensions"]),
        )
        return cls(
            name=data["name"],
            object_type=data["type"],
            transform=transform,
            category=data.get("category", "unknown"),
            is_movable=data.get("is_movable", True),
        )

@dataclass
class RoomBounds:
    """
    Limiti spaziali della stanza, utilizzati per vincolare la randomizzazione
    e validare le coordinate suggerite dall'LLM.

    Attributes:
        x_min: Limite minimo sull'asse X.
        x_max: Limite massimo sull'asse X.
        y_min: Limite minimo sull'asse Y.
        y_max: Limite massimo sull'asse Y.
        z_floor: Quota del pavimento (tipicamente 0.0).
        z_ceiling: Quota del soffitto.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_floor: float = 0.0
    z_ceiling: float = 3.0

    @property
    def width(self) -> float:
        """Larghezza della stanza sull'asse X."""
        return self.x_max - self.x_min

    @property
    def depth(self) -> float:
        """Profondi della stanza sull'asse Y."""
        return self.y_max - self.y_min

    @property
    def height(self) -> float:
        """Altezza della stanza."""
        return self.z_ceiling - self.z_floor

    def clamp_location(self, location: list[float]) -> list[float]:
        """
        Porta le coordinate dentro i bounds della stanza.

        La coordinata Z viene lasciata invariata per non alterare
        la quota di appoggio degli oggetti.

        Args:
            location: Coordinate [x, y, z] da vincolare.

        Returns:
            Coordinate vincolate ai bounds.
        """
        return [
            max(self.x_min, min(self.x_max, location[0])),
            max(self.y_min, min(self.y_max, location[1])),
            location[2],  # Z non viene modificato
        ]

    def to_dict(self) -> dict:
        """Serializza in dizionario compatibile con JSON."""
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_floor": self.z_floor,
            "z_ceiling": self.z_ceiling,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoomBounds":
        """Deserializza da dizionario JSON."""
        return cls(
            x_min=data["x_min"],
            x_max=data["x_max"],
            y_min=data["y_min"],
            y_max=data["y_max"],
            z_floor=data.get("z_floor", 0.0),
            z_ceiling=data.get("z_ceiling", 3.0),
        )

@dataclass
class SceneState:
    """
    Rappresenta lo stato completo di una scena in un determinato momento.

    Attributes:
        scene_name: Nome identificativo della scena.
        objects: Lista degli oggetti presenti nella scena.
        room_bounds: Limiti spaziali della stanza.
        pipeline_step: Etichetta dello stato (es. 'original', 'randomized', 'reordered').
        metadata: Dizionario opzionale per dati aggiuntivi.
    """

    scene_name: str
    objects: list[SceneObject]
    room_bounds: Optional[RoomBounds] = None
    pipeline_step: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def movable_objects(self) -> list[SceneObject]:
        """Restituisce solo gli oggetti marcati come movibili."""
        return [obj for obj in self.objects if obj.is_movable]

    @property
    def static_objects(self) -> list[SceneObject]:
        """Restituisce solo gli oggetti statici (non movibili)."""
        return [obj for obj in self.objects if not obj.is_movable]

    def get_object_by_name(self, name: str) -> Optional[SceneObject]:
        """
        Cerca un oggetto per nome.

        Args:
            name: Nome dell'oggetto da cercare.

        Returns:
            L'oggetto trovato oppure None.
        """
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    def to_dict(self) -> dict:
        """Serializza lo stato completo in dizionario compatibile con JSON."""
        return {
            "scene_name": self.scene_name,
            "pipeline_step": self.pipeline_step,
            "room_bounds": self.room_bounds.to_dict() if self.room_bounds else None,
            "objects": [obj.to_dict() for obj in self.objects],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneState":
        """Deserializza da dizionario JSON."""
        room_bounds = None
        if data.get("room_bounds"):
            room_bounds = RoomBounds.from_dict(data["room_bounds"])

        objects = [SceneObject.from_dict(obj_data) for obj_data in data["objects"]]

        return cls(
            scene_name=data["scene_name"],
            objects=objects,
            room_bounds=room_bounds,
            pipeline_step=data.get("pipeline_step", "unknown"),
            metadata=data.get("metadata", {}),
        )

@dataclass
class LLMCorrection:
    """
    Rappresenta una correzione suggerita dall'LLM Vision.

    Attributes:
        object_name: Nome dell'oggetto da correggere.
        action: Tipo di azione ('move', 'rotate', 'move_and_rotate').
        new_location: Nuova posizione opzionale [x, y, z].
        new_rotation_euler: Nuova rotazione opzionale [rx, ry, rz] in radianti.
        reason: Motivazione testuale fornita dall'LLM.
    """

    object_name: str
    action: str
    new_location: Optional[list[float]] = None
    new_rotation_euler: Optional[list[float]] = None
    reason: str = ""

    def to_dict(self) -> dict:
        """Serializza in dizionario compatibile con JSON."""
        return {
            "object_name": self.object_name,
            "action": self.action,
            "new_location": self.new_location,
            "new_rotation_euler": self.new_rotation_euler,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMCorrection":
        """Deserializza da dizionario JSON."""
        return cls(
            object_name=data["object_name"],
            action=data["action"],
            new_location=data.get("new_location"),
            new_rotation_euler=data.get("new_rotation_euler"),
            reason=data.get("reason", ""),
        )

# Radianti per PI greco, usato nelle conversioni angolari
PI = math.pi

def degrees_to_radians(degrees: float) -> float:
    """Converte gradi in radianti."""
    return degrees * PI / 180.0

def radians_to_degrees(radians: float) -> float:
    """Converte radianti in gradi."""
    return radians * 180.0 / PI
