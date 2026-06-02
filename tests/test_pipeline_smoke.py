"""
Smoke test della pipeline pura dopo il refactor "no categorie mobili":
  - default_classification (solo strutturale/non-mesh = fisso; resto = mobile)
  - compute_room_bounds (geometrico)
  - suggest_grouping (geometrico, opzionale) + apply_manual_parents (manuale)
  - randomizer con padre impostato a mano

NON richiede Blender.  python tests/test_pipeline_smoke.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nl2scene3d.core.classify import (
    apply_manual_parents,
    compute_room_bounds,
    default_classification,
    resolve_classification,
    suggest_grouping,
)
from nl2scene3d.core.models import SceneObject, SceneState, Transform
from nl2scene3d.core.randomizer import SceneRandomizer

_passed = 0
_failed = 0


def check(label, condition):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"[OK  ] {label}")
    else:
        _failed += 1
        print(f"[FAIL] {label}")


def make(name, x, y, z, w, d, h, otype="MESH"):
    cat, mov = default_classification(name, otype, [w, d, h])
    return SceneObject(
        name=name, object_type=otype,
        transform=Transform(location=[x, y, z], rotation_euler=[0, 0, 0], dimensions=[w, d, h]),
        category=cat, is_movable=mov,
    )


# --- Scena sintetica (nomi liberi: niente categorie di mobili) ---
objects = [
    make("structural_room", 0.0,  0.0, 1.35, 6.0, 4.0, 2.7),  # structural -> fisso
    make("Light",           0.0,  0.0, 2.50, 0.1, 0.1, 0.1, otype="LIGHT"),  # non-mesh -> fisso
    make("furniture_bed",  -1.0, -1.0, 0.25, 2.0, 1.6, 0.5),  # object -> mobile
    make("oggetto_strano",  1.0, -1.0, 0.375, 1.2, 0.6, 0.75),  # nome non standard -> mobile
    make("comodino",        1.5,  1.0, 0.25, 0.5, 0.4, 0.5),  # object -> mobile (sara' padre)
    make("lampada_da_tavolo", 1.5, 1.0, 0.70, 0.2, 0.2, 0.4),  # object -> mobile (sopra il comodino)
]
by_name = {o.name: o for o in objects}

# --- default_classification ---
check("structural_room -> structural, fisso",
      (by_name["structural_room"].category, by_name["structural_room"].is_movable) == ("structural", False))
check("Light (non-mesh) -> technical, fisso",
      (by_name["Light"].category, by_name["Light"].is_movable) == ("technical", False))
check("furniture_bed -> object, mobile",
      (by_name["furniture_bed"].category, by_name["furniture_bed"].is_movable) == ("object", True))
check("nome non standard 'oggetto_strano' -> object, mobile (no categorie!)",
      (by_name["oggetto_strano"].category, by_name["oggetto_strano"].is_movable) == ("object", True))
check("'stanza' (IT) -> structural, fisso",
      default_classification("stanza", "MESH", [6, 4, 2.7]) == ("structural", False))

# --- resolve_classification: l'utente forza fisso ---
check("override fixed=True -> non mobile",
      resolve_classification("oggetto_strano", "MESH", [1, 1, 1], {"fixed": True}) == ("object", False))
check("override assente -> automatico",
      resolve_classification("oggetto_strano", "MESH", [1, 1, 1]) == ("object", True))

# --- Confini stanza (geometrici) ---
bounds = compute_room_bounds(objects)
check("bounds X ~ [-3, 3]", math.isclose(bounds.x_min, -3.0) and math.isclose(bounds.x_max, 3.0))
check("bounds Y ~ [-2, 2]", math.isclose(bounds.y_min, -2.0) and math.isclose(bounds.y_max, 2.0))

# Fallback geometrico: nessun nome strutturale -> usa l'impronta piu' grande.
no_struct = [
    make("aaa", 0, 0, 1.0, 5.0, 5.0, 2.0),   # impronta piu' grande
    make("bbb", 0, 0, 0.5, 1.0, 1.0, 1.0),
]
b2 = compute_room_bounds(no_struct)
check("fallback geometrico: bounds dall'oggetto piu' grande (X ~ [-2.5, 2.5])",
      math.isclose(b2.x_min, -2.5) and math.isclose(b2.x_max, 2.5))

# --- suggest_grouping (geometrico, opzionale) ---
mapping = suggest_grouping(objects)
check("suggest_grouping: lampada proposta figlia del comodino",
      mapping.get("lampada_da_tavolo") == "comodino")
check("suggest_grouping: non propone padri per strutturali/fissi",
      "structural_room" not in mapping and "Light" not in mapping)

# --- apply_manual_parents (la scelta dell'utente) ---
apply_manual_parents(objects, {"lampada_da_tavolo": "comodino"})
check("apply_manual_parents: lampada ha padre comodino",
      by_name["lampada_da_tavolo"].parent == "comodino")
check("apply_manual_parents: comodino ha la lampada tra i figli",
      "lampada_da_tavolo" in by_name["comodino"].children)
check("apply_manual_parents: bed resta root",
      by_name["furniture_bed"].parent is None)
# Coppia non valida ignorata.
apply_manual_parents(objects, {"furniture_bed": "inesistente"})
check("apply_manual_parents: padre inesistente ignorato",
      by_name["furniture_bed"].parent is None)

# Ripristina il grouping voluto per il test randomizer.
apply_manual_parents(objects, {"lampada_da_tavolo": "comodino"})

# --- Randomizer: deve restare valido (collisioni/bounds/Z) ---
state = SceneState(scene_name="scena", objects=objects, room_bounds=bounds, pipeline_step="original")
orig_z = {o.name: o.transform.location[2] for o in objects}
randomized = SceneRandomizer(seed=42).randomize(state)
r = {o.name: o for o in randomized.objects}

check("randomize: stesso numero di oggetti", len(randomized.objects) == len(objects))
check("randomize: Z mai modificata",
      all(math.isclose(r[n].transform.location[2], orig_z[n], abs_tol=1e-9) for n in orig_z))
inside = all(
    bounds.contains_aabb(o.transform.aabb_xy(margin=0.0), margin=0.0)
    for o in randomized.objects if o.is_movable
)
check("randomize: oggetti mobili dentro i muri", inside)
check("randomize: structural_room non spostato (fisso)",
      r["structural_room"].transform.location[0] == 0.0)
check("randomize: la lampada segue il comodino (stesso XY)",
      math.isclose(r["comodino"].transform.location[0], r["lampada_da_tavolo"].transform.location[0], abs_tol=1e-6)
      and math.isclose(r["comodino"].transform.location[1], r["lampada_da_tavolo"].transform.location[1], abs_tol=1e-6))

# --- format_inspection ---
from nl2scene3d.core.scene_io import format_inspection
report = format_inspection(state)
check("report contiene il nome scena", "scena" in report)
check("report mostra il padre della lampada", "comodino" in report)

print("-" * 50)
print(f"Totale: {_passed} OK, {_failed} FAIL")
sys.exit(1 if _failed else 0)
