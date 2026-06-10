# nl2scene3d/core/settings.py
"""
Costanti di configurazione di NL2Scene3D.

Filosofia di progetto:
    Niente singleton, niente file TOML, niente variabili d'ambiente.
    Le poche costanti operative vivono qui come dataclass immutabile,
    testabile senza Blender. Il seed del randomizer sta invece nelle
    Addon Preferences perche' e' una scelta dell'utente.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    """Parametri operativi dell'add-on, raggruppati in un'unica struttura immutabile."""

    # --- Geometria / piazzamento ---
    wall_margin:            float = 0.20   # distanza minima dai muri (m)
    collision_margin:       float = 0.05   # gioco extra tra oggetti adiacenti (m)
    jitter_ratio:           float = 0.80   # ampiezza del disordine, come frazione della stanza
    max_placement_attempts: int   = 200    # tentativi massimi per trovare una posizione libera

    # --- Limiti ---
    max_movable_objects: int = 50          # oltre questo limite gli oggetti vengono trattati come fissi

    # --- Rendering ---
    render_edge_px: int = 768              # lato (px) dei render inviati all'LLM


# Istanza unica usata in tutto il package.
CONST = Constants()


# Tipi di oggetto Blender che non sono mai mobili (non-mesh): fissi per default.
NON_MESH_TYPES = frozenset(
    {"CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"}
)


# Parole chiave che identificano elementi strutturali (muri, pavimento, stanza,
# porte, finestre), in italiano e in inglese.
# Usato esclusivamente per:
#   - stimare fissi di default questi elementi;
#   - dedurre i confini della stanza.
# Non esistono categorie per i mobili: quelle le decide l'utente dal pannello.
STRUCTURAL_PATTERNS = [
    "wall", "floor", "ceiling", "room", "door", "window",
    "muro", "parete", "pavimento", "soffitto", "porta", "finestra", "stanza",
]