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
    render_edge_px: int = 1024             # lato (px) dei render inviati all'LLM

    # --- Clearance aperture ---
    door_clearance_depth:   float = 0.90   # profondita' zona di rispetto davanti a una porta (m)
    window_clearance_depth: float = 0.50   # profondita' zona di rispetto davanti a una finestra (m)

    # --- Punteggi collisione aperture ---
    door_penalty:   float = 50.0           # penalita' collision_score per blocco porta
    window_penalty: float = 25.0           # penalita' collision_score per copertura finestra

    # --- Soglie Z collisioni ---
    wall_collision_z_threshold:      float = 0.01  # sovrapposizione Z minima per check muro (m)
    furniture_collision_z_threshold: float = 0.01  # sovrapposizione Z minima tra mobili (m)

    # --- Parametri MTV / risolutore collisioni ---
    resolve_collisions_max_iter: int   = 80        # iterazioni massime del risolutore MTV
    mtv_buffer:                  float = 0.01      # buffer aggiuntivo post-MTV (m)

    # --- Rendering overlay ---
    scale_bar_target_px:       int   = 140  # larghezza target della barra di scala (px)
    compass_anchor_offset_x:   int   = 92   # offset X bussola dal bordo destro (px)
    compass_anchor_offset_y:   int   = 92   # offset Y bussola dal bordo superiore (px)

    # --- Grouping ---
    grouping_volume_ratio:    float = 1.20  # rapporto volume minimo padre/figlio
    grouping_footprint_ratio: float = 1.05  # rapporto impronta XY minimo padre/figlio
    grouping_on_top_lo:       float = 0.00  # margine inferiore soglia "sopra" (m); 0 = no penetrazione
    grouping_on_top_hi:       float = 0.20  # margine superiore soglia "sopra" (m)
    grouping_z_overlap_frac:  float = 0.30  # frazione minima di sovrapposizione Z per "vicino in Z"

    # --- apply_state ---
    apply_state_tolerance: float = 0.001    # spostamento minimo sotto cui non si aggiorna (m)

    # --- collision_score post-LLM ---
    post_llm_check_margin: float = 0.02     # margine OBB nel check post-LLM (m)


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