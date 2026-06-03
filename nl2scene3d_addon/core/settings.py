# nl2scene3d/core/settings.py
"""
Costanti di NL2Scene3D (sostituisce il vecchio config.py + settings.toml).

Filosofia: niente singleton, niente TOML, niente .env. Le poche manopole reali
vivono qui come costanti pure (testabili senza Blender). Il seed del randomizer
sta invece nelle Addon Preferences, perche' lo sceglie l'utente.
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

    # --- Limiti ---
    max_movable_objects: int = 50        # oltre questo numero gli oggetti restano fissi

    # --- Rendering (usato dallo Step 4) ---
    render_edge_px: int = 768            # lato del render inviato all'LLM (= no resize lato LLM)


# Istanza unica usata in tutto il package.
CONST = Constants()


# Tipi Blender mai mobili (non-mesh): fissi di default.
NON_MESH_TYPES = frozenset(
    {"CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"}
)

# Piccolo insieme di parole che identificano elementi STRUTTURALI (muri/pavimento/
# stanza/porte/finestre), multilingua IT/EN. Serve SOLO a:
#   - stimare fissi di default questi elementi (comportamento che l'utente apprezza);
#   - definire i confini della stanza.
# NON esistono piu' categorie per i mobili: quelle le decide l'utente (fisso/mobile
# e padre-figlio) dal pannello.
STRUCTURAL_PATTERNS = [
    "wall", "floor", "ceiling", "room", "door", "window",
    "muro", "parete", "pavimento", "soffitto", "porta", "finestra", "stanza",
]
