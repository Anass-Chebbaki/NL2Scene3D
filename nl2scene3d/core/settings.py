# nl2scene3d/core/settings.py
"""
Costanti di NL2Scene3D (sostituisce il vecchio config.py + settings.toml).

Filosofia: niente singleton, niente TOML, niente .env. Le poche manopole reali
vivono qui come costanti pure (testabili senza Blender). Le impostazioni
specifiche del backend (modello Ollama, URL, temperature) e il seed stanno
invece nelle Addon Preferences / nel pannello, perche' le sceglie l'utente.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    # --- Geometria / piazzamento ---
    wall_margin: float = 0.20            # distanza minima dai muri (m)
    collision_margin: float = 0.05       # gioco extra tra oggetti (m)
    jitter_ratio: float = 0.80           # ampiezza del disordine, frazione della stanza
    max_placement_attempts: int = 200    # tentativi per trovare una posizione libera

    # --- Classificazione oggetti ---
    max_movable_objects: int = 50        # oltre questo numero gli oggetti restano fissi
    min_object_dimension: float = 0.05   # oggetti piu' piccoli = decorazioni fisse (m)
    static_height_threshold: float = 1.0 # oggetti con base >= a questa quota = fissi (m)
    freeze_ceiling_objects: bool = True  # congela anche gli oggetti a soffitto

    # --- Rendering (usato dallo Step 4) ---
    render_edge_px: int = 768            # lato del render inviato all'LLM (= no resize lato Ollama)


# Istanza unica usata in tutto il package.
CONST = Constants()


# Tipi Blender mai mobili (non-mesh).
NON_MESH_TYPES = frozenset(
    {"CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"}
)

# Pattern di nome che identificano elementi strutturali (multilingua IT/EN).
STRUCTURAL_PATTERNS = [
    "wall", "floor", "ceiling", "room", "door", "window",
    "muro", "parete", "pavimento", "soffitto", "porta", "finestra", "stanza",
]

# Pattern di luci a soffitto (restano fisse) - IT/EN.
CEILING_LIGHT_PATTERNS = ["ceiling", "pendant", "chandelier", "soffitto", "plafoniera", "lampadario"]
