"""
Smoke test del nucleo puro (models + geometry) di NL2Scene3D.

NON richiede Blender: verifica la geometria e i modelli dati che sono il vero
cuore dell'add-on. Eseguire dalla root del package nuovo:

    python tests/test_core_smoke.py

Stampa una riga per ogni controllo e un riepilogo finale.
"""

import math
import sys
from pathlib import Path

# Rende importabile il package 'nl2scene3d' che sta nella cartella accanto a tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nl2scene3d.core.models import RoomBounds, SceneObject, SceneState, Transform
from nl2scene3d.core import geometry as geo

_passed = 0
_failed = 0


def check(label: str, condition: bool) -> None:
    global _passed, _failed
    status = "OK  " if condition else "FAIL"
    if condition:
        _passed += 1
    else:
        _failed += 1
    print(f"[{status}] {label}")


def box(name, x, y, w=1.0, d=1.0, h=1.0, rz=0.0, category="furniture", movable=True):
    """Crea un SceneObject parallelepipedo centrato in (x, y) con rotazione rz (rad)."""
    return SceneObject(
        name=name,
        object_type="MESH",
        transform=Transform(
            location=[x, y, h / 2.0],
            rotation_euler=[0.0, 0.0, rz],
            dimensions=[w, d, h],
        ),
        category=category,
        is_movable=movable,
    )


# ---------------------------------------------------------------------------
# Transform: AABB, OBB, centro, z_range
# ---------------------------------------------------------------------------

t = Transform(location=[0, 0, 0.5], rotation_euler=[0, 0, 0], dimensions=[2.0, 4.0, 1.0])
xmin, xmax, ymin, ymax = t.aabb_xy()
check("AABB non ruotato: larghezza X = 2", math.isclose(xmax - xmin, 2.0))
check("AABB non ruotato: profondita' Y = 4", math.isclose(ymax - ymin, 4.0))

t90 = Transform(location=[0, 0, 0.5], rotation_euler=[0, 0, math.pi / 2], dimensions=[2.0, 4.0, 1.0])
xmin2, xmax2, ymin2, ymax2 = t90.aabb_xy()
check("AABB ruotato 90deg: X e Y si scambiano (X~4)", math.isclose(xmax2 - xmin2, 4.0, abs_tol=1e-9))
check("AABB ruotato 90deg: Y~2", math.isclose(ymax2 - ymin2, 2.0, abs_tol=1e-9))

zmin, zmax = t.z_range()
check("z_range: base a 0, top a 1", math.isclose(zmin, 0.0) and math.isclose(zmax, 1.0))

cx, cy = t.geometric_center_xy()
check("centro geometrico senza offset = location", math.isclose(cx, 0.0) and math.isclose(cy, 0.0))

corners = t.obb_corners_xy()
check("OBB: 4 angoli", len(corners) == 4)


# ---------------------------------------------------------------------------
# RoomBounds: clamp e contains
# ---------------------------------------------------------------------------

room = RoomBounds(x_min=-3.0, x_max=3.0, y_min=-2.0, y_max=2.0, z_floor=0.0, z_ceiling=2.7)
check("RoomBounds.width", math.isclose(room.width, 6.0))
check("RoomBounds.height", math.isclose(room.height, 2.7))

clamped = room.clamp_location([10.0, 0.0, 0.5], dimensions=[1.0, 1.0, 1.0], margin=0.2)
check("clamp_location riporta X dentro i muri", clamped[0] <= 3.0 - 0.2 - 0.5 + 1e-9)
check("clamp_location non tocca Z", math.isclose(clamped[2], 0.5))

check("contains_aabb: oggetto interno = True", room.contains_aabb((-1, 1, -1, 1)))
check("contains_aabb: oggetto sporgente = False", not room.contains_aabb((-1, 5, -1, 1)))


# ---------------------------------------------------------------------------
# Geometry: SAT, collisioni, MTV
# ---------------------------------------------------------------------------

a = box("A", 0.0, 0.0)
b_overlap = box("B", 0.5, 0.0)   # si sovrappone ad A
b_far     = box("C", 5.0, 5.0)   # lontano

check("SAT: due box sovrapposti -> overlap",
      geo.sat_overlap(a.transform.obb_corners_xy(), b_overlap.transform.obb_corners_xy()))
check("SAT: due box lontani -> no overlap",
      not geo.sat_overlap(a.transform.obb_corners_xy(), b_far.transform.obb_corners_xy()))

check("has_collision: A vs [B sovrapposto] -> True",
      geo.has_collision(a, [b_overlap], check_walls=False))
check("has_collision: A vs [C lontano] -> False",
      not geo.has_collision(a, [b_far], check_walls=False))

dx, dy = geo.penetration_vector(a, b_overlap, margin=0.0)
moved_a = a.copy()
moved_a.transform.location[0] += dx
moved_a.transform.location[1] += dy
# Nota: si verifica con lo STESSO margine usato per risolvere (0.0). Se qui si
# usasse il furniture_margin di default (0.05) gli OBB verrebbero gonfiati di
# 5 cm per lato e risulterebbero ancora "in collisione": e' il motivo per cui il
# solver reale usa margini coerenti tra spinta e verifica e itera piu' volte.
check("penetration_vector: dopo lo spostamento niente piu' collisione (stesso margine)",
      not geo.has_collision(moved_a, [b_overlap], check_walls=False, furniture_margin=0.0))

check("penetration_vector: nessuna sovrapposizione -> (0,0)",
      geo.penetration_vector(a, b_far) == (0.0, 0.0))

# Contenimento nei confini stanza (check muri "virtuali").
inside  = box("inside", 0.0, 0.0, w=1.0, d=1.0)
outside = box("outside", 2.9, 0.0, w=1.0, d=1.0)  # sfora il muro X a margine 0.2
check("has_collision con room_bounds: oggetto interno -> False",
      not geo.has_collision(inside, [], wall_margin=0.2, room_bounds=room))
check("has_collision con room_bounds: oggetto a muro -> True",
      geo.has_collision(outside, [], wall_margin=0.2, room_bounds=room))


# ---------------------------------------------------------------------------
# snap_rotation_90
# ---------------------------------------------------------------------------

check("snap 80deg -> 90deg", math.isclose(geo.snap_rotation_90(math.radians(80)), math.pi / 2))
check("snap 10deg -> 0deg", math.isclose(geo.snap_rotation_90(math.radians(10)), 0.0))


# ---------------------------------------------------------------------------
# group_aabb_xy: il figlio allarga l'AABB del gruppo
# ---------------------------------------------------------------------------

parent = box("parent", 0.0, 0.0, w=1.0, d=1.0)
child  = box("child",  1.0, 0.0, w=1.0, d=1.0)
gx_min, gx_max, gy_min, gy_max = geo.group_aabb_xy(
    parent, parent.transform.location, 0.0, [child]
)
check("group_aabb_xy: il gruppo padre+figlio e' largo ~2 in X", math.isclose(gx_max - gx_min, 2.0))


# ---------------------------------------------------------------------------
# SceneState: round-trip to_dict/from_dict
# ---------------------------------------------------------------------------

state = SceneState(
    scene_name="bedroom",
    objects=[parent, child],
    room_bounds=room,
    pipeline_step="original",
)
restored = SceneState.from_dict(state.to_dict())
check("SceneState round-trip: stesso numero di oggetti", len(restored.objects) == 2)
check("SceneState round-trip: room_bounds preservato", math.isclose(restored.room_bounds.width, 6.0))
check("SceneState.get() trova per nome", restored.get("child") is not None)
check("SceneState.movable_objects", len(state.movable_objects) == 2)


# ---------------------------------------------------------------------------
# Riepilogo
# ---------------------------------------------------------------------------

print("-" * 50)
print(f"Totale: {_passed} OK, {_failed} FAIL")
sys.exit(1 if _failed else 0)
