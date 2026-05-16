# nl2scene3d/scene_loader.py
"""
Shim di compatibilità — l'implementazione reale è in scene_state.py.

Questo file esiste solo per non rompere il codice che importa:
    from nl2scene3d.scene_loader import SceneLoader

Tutto risiede ora in scene_state.py, che è il modulo canonico
con classify_object, compute_room_bounds, compute_grouping e SceneLoader.
"""
from nl2scene3d.scene_state import (  # noqa: F401
    SceneLoader,
    classify_object,
    compute_room_bounds,
    compute_grouping,
)

# Alias dei nomi di metodo vecchi → nuovi, per chi usa il loader direttamente.
# SceneLoader.save_state_to_json  → SceneLoader.save_state
# SceneLoader.load_state_from_json → SceneLoader.load_state
# (Questi alias vengono aggiunti dinamicamente se necessario.)
if not hasattr(SceneLoader, "save_state_to_json"):
    SceneLoader.save_state_to_json = SceneLoader.save_state  # type: ignore[attr-defined]
if not hasattr(SceneLoader, "load_state_from_json"):
    SceneLoader.load_state_from_json = SceneLoader.load_state  # type: ignore[attr-defined]