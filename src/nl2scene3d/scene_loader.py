"""
Caricamento e introspezione di scene Blender.

Questo modulo implementa:
1. Apertura della scena .blend tramite bpy
2. Estrazione dello stato degli oggetti in formato SceneState

NOTA IMPORTANTE: Questo modulo e' progettato per essere eseguito
all'interno del Python integrato in Blender (bpy disponibile).
Per i test unitari, viene usata una implementazione mock di bpy.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from nl2scene3d.models import (
    ObjectTransform,
    RoomBounds,
    SceneObject,
    SceneState,
)

logger = logging.getLogger(__name__)

# Tipi di oggetti Blender che vengono sempre ignorati durante l'estrazione
# perche' non fanno parte del layout arredativo.
NON_MESH_TYPES: frozenset[str] = frozenset({"CAMERA", "LIGHT", "SPEAKER", "ARMATURE"})

# Nomi (o parti di nomi) che identificano elementi strutturali non movibili.
# Il confronto e' case-insensitive.
STRUCTURAL_NAME_PATTERNS: frozenset[str] = frozenset(
    {
        "wall",
        "floor",
        "ceiling",
        "room",
        "baseboard",
        "muro",
        "pavimento",
        "soffitto",
        "stanza",
        "door",
        "window",
        "porta",
        "finestra",
    }
)

# Nomi di categorie per oggetti strutturali
STRUCTURAL_CATEGORY: str = "structural"

# Dimensione minima (in metri) affinche' un oggetto sia considerato
# abbastanza grande da essere incluso nel layout.
MIN_OBJECT_DIMENSION: float = 0.05

def _classify_object(
    name: str,
    object_type: str,
    dimensions: list[float],
) -> tuple[str, bool]:
    """
    Determina la categoria e la movibilita' di un oggetto.

    Args:
        name: Nome dell'oggetto nella scena Blender.
        object_type: Tipo Blender dell'oggetto.
        dimensions: Dimensioni [x, y, z] del bounding box.

    Returns:
        Tupla (categoria, is_movable).
    """
    name_lower = name.lower()

    # Oggetti non-mesh (camere, luci, ecc.) sono sempre non movibili
    if object_type in NON_MESH_TYPES:
        return "technical", False

    # Oggetti troppo piccoli vengono ignorati (es. viti, chiodi)
    max_dim = max(dimensions) if dimensions else 0.0
    if max_dim < MIN_OBJECT_DIMENSION:
        return "decoration_small", False

    # Le lampade hanno la precedenza sui pattern strutturali perche' 
    # spesso contengono la parola "floor" o "wall" nel nome.
    if any(kw in name_lower for kw in ("lamp", "lampada", "light")):
        # Le lampade da terra sono movibili, quelle a soffitto o muro no
        if any(kw in name_lower for kw in ("ceiling", "soffitto", "pendant", "wall", "muro")):
            return "light_ceiling", False
        return "light_floor", True

    # Elementi strutturali identificati per nome
    for pattern in STRUCTURAL_NAME_PATTERNS:
        if pattern in name_lower:
            return STRUCTURAL_CATEGORY, False

    # Classificazione euristica per categoria di arredo
    if any(kw in name_lower for kw in ("sofa", "couch", "divano")):
        return "seating_large", True
    if any(kw in name_lower for kw in ("chair", "sedia", "stool", "sgabello")):
        return "seating_small", True
    if any(kw in name_lower for kw in ("table", "tavolo", "desk", "scrivania")):
        return "table", True
    if any(kw in name_lower for kw in ("shelf", "scaffale", "bookcase", "libreria")):
        return "storage", True
    if any(kw in name_lower for kw in ("bed", "letto", "mattress", "materasso")):
        return "bed", True
    if any(kw in name_lower for kw in ("rug", "tappeto", "carpet")):
        return "rug", True
    if any(kw in name_lower for kw in ("plant", "pianta", "vase", "vaso")):
        return "decoration", True

    return "furniture", True

def extract_room_bounds_from_objects(
    objects: list[SceneObject],
) -> RoomBounds:
    """
    Calcola i bounds della stanza a partire dagli oggetti strutturali.

    Se non ci sono oggetti strutturali, stima i bounds dai movabili.

    Args:
        objects: Lista completa degli oggetti della scena.

    Returns:
        RoomBounds calcolati.
    """
    structural = [obj for obj in objects if not obj.is_movable]
    if not structural:
        structural = objects

    all_x = [obj.transform.location[0] for obj in structural]
    all_y = [obj.transform.location[1] for obj in structural]
    all_z = [obj.transform.location[2] for obj in structural]

    # Calcolo z_ceiling con euristica sulle dimensioni
    z_with_dims = [
        obj.transform.location[2] + obj.transform.dimensions[2]
        for obj in structural
        if obj.transform.dimensions[2] > 0.1
    ]

    padding = 0.2  # margine di sicurezza in metri

    return RoomBounds(
        x_min=min(all_x) - padding if all_x else -5.0,
        x_max=max(all_x) + padding if all_x else 5.0,
        y_min=min(all_y) - padding if all_y else -5.0,
        y_max=max(all_y) + padding if all_y else 5.0,
        z_floor=min(all_z) if all_z else 0.0,
        z_ceiling=max(z_with_dims) if z_with_dims else 3.0,
    )

class SceneLoader:
    """
    Carica e ispeziona scene Blender tramite l'API bpy.

    Questo loader viene istanziato e usato all'interno dell'ambiente
    Python di Blender, dove bpy e' disponibile come modulo built-in.

    Attributes:
        max_objects: Limite massimo di oggetti movibili da includere.
    """

    def __init__(self, max_objects: int = 20) -> None:
        """
        Inizializza il loader.

        Args:
            max_objects: Numero massimo di oggetti movibili da includere.
                         Valore dal design document: 20.
        """
        self.max_objects = max_objects
        logger.info(
            "SceneLoader inizializzato. Limite oggetti movibili: %d", max_objects
        )

    def load_blend_file(self, blend_path: Path) -> None:
        """
        Apre un file .blend in Blender sostituendo la scena corrente.

        Args:
            blend_path: Percorso al file .blend da aprire.

        Raises:
            FileNotFoundError: Se il file .blend non esiste.
            ImportError: Se bpy non e' disponibile (esecuzione fuori Blender).
        """
        if not blend_path.exists():
            raise FileNotFoundError(
                f"File .blend non trovato: {blend_path}"
            )

        try:
            import bpy  # noqa: PLC0415 - import locale necessario per bpy
        except ImportError as exc:
            raise ImportError(
                "Il modulo 'bpy' non e' disponibile. "
                "Questo metodo deve essere eseguito all'interno di Blender."
            ) from exc

        logger.info("Apertura file .blend: %s", blend_path)
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        logger.info("File .blend aperto con successo.")

    def extract_scene_state(
        self,
        scene_name: Optional[str] = None,
    ) -> SceneState:
        """
        Estrae lo stato corrente della scena aperta in Blender.

        Itera su tutti gli oggetti della scena corrente, li classifica
        e produce un oggetto SceneState serializzabile.

        Args:
            scene_name: Nome identificativo da assegnare alla scena estratta.
                        Se None, usa il nome del file .blend.

        Returns:
            SceneState con tutti gli oggetti classificati.

        Raises:
            ImportError: Se bpy non e' disponibile.
        """
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Il modulo 'bpy' non e' disponibile. "
                "Questo metodo deve essere eseguito all'interno di Blender."
            ) from exc

        blender_scene = bpy.context.scene
        effective_name = scene_name or blender_scene.name

        logger.info(
            "Estrazione stato scena '%s'. Oggetti presenti: %d",
            effective_name,
            len(blender_scene.objects),
        )

        objects: list[SceneObject] = []
        movable_count = 0

        for blender_obj in blender_scene.objects:
            obj_name = blender_obj.name
            obj_type = blender_obj.type

            # Ottiene dimensioni del bounding box in scala applicata
            dimensions = [
                blender_obj.dimensions.x,
                blender_obj.dimensions.y,
                blender_obj.dimensions.z,
            ]

            category, is_movable = _classify_object(obj_name, obj_type, dimensions)

            # Applica il limite al numero di oggetti movibili
            if is_movable and movable_count >= self.max_objects:
                logger.debug(
                    "Oggetto '%s' ignorato: limite di %d oggetti movibili raggiunto.",
                    obj_name,
                    self.max_objects,
                )
                is_movable = False

            transform = ObjectTransform(
                location=[
                    blender_obj.location.x,
                    blender_obj.location.y,
                    blender_obj.location.z,
                ],
                rotation_euler=[
                    blender_obj.rotation_euler.x,
                    blender_obj.rotation_euler.y,
                    blender_obj.rotation_euler.z,
                ],
                dimensions=dimensions,
            )

            scene_obj = SceneObject(
                name=obj_name,
                object_type=obj_type,
                transform=transform,
                category=category,
                is_movable=is_movable,
            )
            objects.append(scene_obj)

            if is_movable:
                movable_count += 1

        logger.info(
            "Estratti %d oggetti totali, di cui %d movibili.",
            len(objects),
            movable_count,
        )

        # Calcolo automatico dei bounds della stanza
        room_bounds = extract_room_bounds_from_objects(objects)
        logger.info(
            "Bounds calcolati: X[%.2f, %.2f] Y[%.2f, %.2f] Z[%.2f, %.2f]",
            room_bounds.x_min,
            room_bounds.x_max,
            room_bounds.y_min,
            room_bounds.y_max,
            room_bounds.z_floor,
            room_bounds.z_ceiling,
        )

        return SceneState(
            scene_name=effective_name,
            objects=objects,
            room_bounds=room_bounds,
            pipeline_step="original",
        )

    def save_state_to_json(
        self,
        state: SceneState,
        output_path: Path,
    ) -> None:
        """
        Serializza uno SceneState in un file JSON.

        Args:
            state: Stato della scena da salvare.
            output_path: Percorso del file JSON di destinazione.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)

        logger.info(
            "Stato scena '%s' (tag: %s) salvato in: %s",
            state.scene_name,
            state.pipeline_step,
            output_path,
        )

    @staticmethod
    def load_state_from_json(json_path: Path) -> SceneState:
        """
        Carica uno SceneState da un file JSON.

        Args:
            json_path: Percorso al file JSON da leggere.

        Returns:
            SceneState deserializzato.

        Raises:
            FileNotFoundError: Se il file JSON non esiste.
            json.JSONDecodeError: Se il file non e' JSON valido.
        """
        if not json_path.exists():
            raise FileNotFoundError(
                f"File JSON non trovato: {json_path}"
            )

        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        state = SceneState.from_dict(data)
        logger.info(
            "Stato scena '%s' (tag: %s) caricato da: %s",
            state.scene_name,
            state.pipeline_step,
            json_path,
        )
        return state
