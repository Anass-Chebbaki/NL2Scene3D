# src/nl2scene3d/scene_loader.py
"""
Caricamento e introspezione di scene Blender.

Implementa:
1. Apertura della scena .blend tramite bpy
2. Estrazione dello stato degli oggetti in formato SceneState
3. Serializzazione e deserializzazione dello stato in JSON

Questo modulo e' progettato per essere eseguito all'interno del Python
integrato in Blender (bpy disponibile). Per i test unitari viene usata
una implementazione mock di bpy.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from nl2scene3d.models import (
    ObjectTransform,
    RoomBounds,
    SceneObject,
    SceneState,
)
from nl2scene3d.config import PipelineConfig

logger = logging.getLogger(__name__)

STRUCTURAL_CATEGORY: str = "structural"


def extract_room_bounds_from_objects(
    objects: list[SceneObject],
) -> RoomBounds:
    """
    Calcola i bounds della stanza dagli oggetti strutturali.

    Strategia:
    1. Se esiste un singolo oggetto strutturale "grande" (volume > 50% del
       volume totale strutturale), usa le sue dimensioni come stanza.
       Questo gestisce le scene Sketchfab dove la stanza e' un'unica mesh.
    2. Altrimenti combina gli AABB di tutti gli oggetti strutturali
       (caso classico di muri separati).
    3. Per Z usa sempre un range realistico (0 - 2.5m) per evitare
       che oggetti siano randomizzati troppo in alto.

    Args:
        objects: Lista completa degli oggetti della scena.

    Returns:
        RoomBounds calcolati con margini realistici.
    """
    structural = [obj for obj in objects if not obj.is_movable]

    if not structural:
        logger.warning(
            "Nessun oggetto strutturale trovato. "
            "I bounds della stanza vengono stimati dall'insieme completo degli oggetti."
        )
        structural = objects

    if not structural:
        logger.warning(
            "La scena non contiene oggetti. Vengono usati bounds di default."
        )
        return RoomBounds(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0)

    # Strategia 1: cerca un'unica mesh-stanza grande (es. structural_room).
    main_room: Optional[SceneObject] = None
    for obj in structural:
        # Volume dell'oggetto
        vol = (
            obj.transform.dimensions[0]
            * obj.transform.dimensions[1]
            * obj.transform.dimensions[2]
        )
        # Soglia minima: oggetto > 5x5x2 metri = 50 m^3
        if vol > 50.0 and "room" in obj.name.lower():
            main_room = obj
            break

    if main_room is not None:
        logger.info(
            "Stanza identificata da oggetto unico: '%s' (dim: %.2fx%.2fx%.2f)",
            main_room.name,
            main_room.transform.dimensions[0],
            main_room.transform.dimensions[1],
            main_room.transform.dimensions[2],
        )
        loc = main_room.transform.location
        dim = main_room.transform.dimensions
        return RoomBounds(
            x_min=loc[0] - dim[0] / 2.0,
            x_max=loc[0] + dim[0] / 2.0,
            y_min=loc[1] - dim[1] / 2.0,
            y_max=loc[1] + dim[1] / 2.0,
            z_floor=0.0,
            z_ceiling=2.5,
        )

    # Strategia 2: combina AABB di tutti gli oggetti strutturali (muri separati).
    all_x_min = [
        obj.transform.location[0] - obj.transform.dimensions[0] / 2.0
        for obj in structural
    ]
    all_x_max = [
        obj.transform.location[0] + obj.transform.dimensions[0] / 2.0
        for obj in structural
    ]
    all_y_min = [
        obj.transform.location[1] - obj.transform.dimensions[1] / 2.0
        for obj in structural
    ]
    all_y_max = [
        obj.transform.location[1] + obj.transform.dimensions[1] / 2.0
        for obj in structural
    ]

    return RoomBounds(
        x_min=min(all_x_min),
        x_max=max(all_x_max),
        y_min=min(all_y_min),
        y_max=max(all_y_max),
        z_floor=0.0,
        z_ceiling=2.5,
    )


class SceneLoader:
    """
    Carica e ispeziona scene Blender tramite l'API bpy.

    Deve essere istanziato e usato all'interno dell'ambiente Python di
    Blender, dove bpy e' disponibile come modulo built-in.

    Attributes:
        config: Configurazione della pipeline.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Inizializza il loader.

        Args:
            config: Configurazione della pipeline.
        """
        self.config = config
        logger.info(
            "SceneLoader inizializzato. Limite oggetti movibili: %d.",
            config.max_movable_objects,
        )

    def _classify_object(
        self,
        name: str,
        object_type: str,
        dimensions: list[float],
    ) -> tuple[str, bool]:
        """
        Determina la categoria e la movibilita' di un oggetto.

        Usa i pattern di classificazione definiti nella configurazione.

        Args:
            name: Nome dell'oggetto nella scena Blender.
            object_type: Tipo Blender dell'oggetto.
            dimensions: Dimensioni [x, y, z] del bounding box.

        Returns:
            Tupla (categoria, is_movable).
        """
        name_lower = name.lower()

        if object_type in self.config.non_mesh_types:
            return "technical", False

        max_dim = max(dimensions) if dimensions else 0.0
        if max_dim < self.config.min_object_dimension:
            return "decoration_small", False

        if any(kw in name_lower for kw in ("lamp", "lampada", "light")):
            if any(kw in name_lower for kw in self.config.ceiling_light_patterns):
                return "light_ceiling", False
            return "light_floor", True

        for pattern in self.config.structural_patterns:
            if pattern in name_lower:
                return STRUCTURAL_CATEGORY, False

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

    def load_blend_file(self, blend_path: Path) -> None:
        """
        Apre un file .blend in Blender sostituendo la scena corrente.

        Args:
            blend_path: Percorso al file .blend da aprire.

        Raises:
            FileNotFoundError: Se il file .blend non esiste.
            ImportError: Se bpy non e' disponibile.
        """
        if not blend_path.exists():
            raise FileNotFoundError(
                f"File .blend non trovato: {blend_path}"
            )

        try:
            import bpy  # noqa: PLC0415
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
                        Se None, usa il nome della scena Blender corrente.

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
            "Estrazione stato scena '%s'. Oggetti presenti: %d.",
            effective_name,
            len(blender_scene.objects),
        )

        objects: list[SceneObject] = []
        movable_count = 0

        for blender_obj in blender_scene.objects:
            obj_name: str = blender_obj.name
            obj_type: str = blender_obj.type

            dimensions = [
                blender_obj.dimensions.x,
                blender_obj.dimensions.y,
                blender_obj.dimensions.z,
            ]

            category, is_movable = self._classify_object(
                obj_name, obj_type, dimensions
            )

            if is_movable and movable_count >= self.config.max_movable_objects:
                logger.debug(
                    "Oggetto '%s' declassato a non movibile: "
                    "limite di %d oggetti movibili raggiunto.",
                    obj_name,
                    self.config.max_movable_objects,
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

        room_bounds = extract_room_bounds_from_objects(objects)

        logger.info(
            "Bounds calcolati: X[%.2f, %.2f] Y[%.2f, %.2f] Z[%.2f, %.2f].",
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
            "Stato scena '%s' (step: %s) salvato in: %s",
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
            "Stato scena '%s' (step: %s) caricato da: %s",
            state.scene_name,
            state.pipeline_step,
            json_path,
        )
        return state