# tests/test_core.py
import sys
import os
import unittest
import math

# Aggiunge il path del workspace per poter importare nl2scene3d_addon
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nl2scene3d_addon.core.models import SceneObject, Transform, RoomBounds, SceneState
from nl2scene3d_addon.core.geometry import sat_overlap, has_collision, collision_score
from nl2scene3d_addon.core.classify import compute_room_bounds, default_classification
from nl2scene3d_addon.core.reorganizer import extract_json, sanitize_response, _contains_key_recursive

class TestGeometry(unittest.TestCase):
    def test_sat_overlap_basic(self):
        # Due box allineati agli assi che si sovrappongono
        poly1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
        poly2 = [(1, 1), (3, 1), (3, 3), (1, 3)]
        self.assertTrue(sat_overlap(poly1, poly2))

    def test_sat_overlap_no_overlap(self):
        # Due box allineati senza sovrapposizione
        poly1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly2 = [(2, 2), (3, 2), (3, 3), (2, 3)]
        self.assertFalse(sat_overlap(poly1, poly2))

    def test_sat_overlap_rotated(self):
        # Box ruotati
        # poly1: quadrato 2x2 centrato in (0,0), non ruotato
        poly1 = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        # poly2: quadrato ruotato di 45 gradi posizionato a (1.4, 0)
        # Angolo a 45 gradi ha cos=sin=~0.707. Lato=2, semi-lato=1.
        # Vertici locali: (-1,-1), (1,-1), (1,1), (-1,1)
        # Ruotato e traslato di 1.4 su X:
        poly2 = [
            (1.4 - 1.414, 0.0),
            (1.4, -1.414),
            (1.4 + 1.414, 0.0),
            (1.4, 1.414)
        ]
        self.assertTrue(sat_overlap(poly1, poly2))

        # Spostato piu' lontano: nessuna collisione
        poly3 = [
            (2.5 - 1.414, 0.0),
            (2.5, -1.414),
            (2.5 + 1.414, 0.0),
            (2.5, 1.414)
        ]
        self.assertFalse(sat_overlap(poly1, poly3))


class TestClassify(unittest.TestCase):
    def test_default_classification(self):
        cat, mov = default_classification("wall_north", "MESH", [5.0, 0.2, 2.7])
        self.assertEqual(cat, "structural")
        self.assertFalse(mov)

        cat2, mov2 = default_classification("table_wood", "MESH", [1.2, 0.8, 0.75])
        self.assertEqual(cat2, "object")
        self.assertTrue(mov2)

    def test_compute_room_bounds(self):
        # Stanza definita da un pavimento strutturale grande
        floor = SceneObject(
            name="floor",
            object_type="MESH",
            transform=Transform(
                location=[0.0, 0.0, 0.0],
                rotation_euler=[0.0, 0.0, 0.0],
                dimensions=[10.0, 8.0, 0.1]
            ),
            category="structural",
            is_movable=False
        )
        objects = [floor]
        bounds = compute_room_bounds(objects)
        self.assertAlmostEqual(bounds.x_min, -5.0)
        self.assertAlmostEqual(bounds.x_max, 5.0)
        self.assertAlmostEqual(bounds.y_min, -4.0)
        self.assertAlmostEqual(bounds.y_max, 4.0)


class TestReorganizer(unittest.TestCase):
    def test_extract_json_metadata_preamble(self):
        text = """
        Some preamble text before the JSON block.
        {
           "metadata": {"model": "gemini-3.5", "tokens": 1234}
        }
        Now some other text, and then the real placements:
        ```json
        {
           "placements": [
              {"name": "table_wood", "x": 1.0, "y": 2.0, "rotation_deg": 90.0}
           ]
        }
        ```
        """
        parsed = extract_json(text)
        self.assertIsNotNone(parsed)
        self.assertIn("placements", parsed)
        self.assertEqual(len(parsed["placements"]), 1)

    def test_sanitize_response_rotation_drift(self):
        # Scenario: Un gruppo con un Parent e un Child.
        # Il parent viene ruotato di 90 gradi dall'LLM.
        # Verifichiamo che l'offset locale del figlio rispetto al parent
        # ruotato si applichi correttamente senza deriva delle posizioni.
        orig_parent = SceneObject(
            name="parent_desk",
            object_type="MESH",
            transform=Transform(
                location=[0.0, 0.0, 0.0],
                rotation_euler=[0.0, 0.0, 0.0],  # 0 gradi
                dimensions=[2.0, 1.0, 0.75]
            ),
            category="object",
            is_movable=True,
            children=["child_lamp"]
        )
        orig_child = SceneObject(
            name="child_lamp",
            object_type="MESH",
            transform=Transform(
                location=[0.5, 0.0, 0.8],  # A destra del centro del tavolo (+0.5 su X)
                rotation_euler=[0.0, 0.0, 0.0],
                dimensions=[0.3, 0.3, 0.4]
            ),
            category="object",
            is_movable=True,
            parent="parent_desk"
        )
        
        state = SceneState(
            scene_name="TestScene",
            objects=[orig_parent, orig_child],
            room_bounds=RoomBounds(-5, 5, -5, 5, 0, 3)
        )
        
        # Proposta LLM: sposta il tavolo a (1.0, 1.0) e lo ruota di 90 gradi (pi/2)
        # 90 gradi in radianti = 1.57079632679
        llm_response = {
            "placements": [
                {
                    "name": "parent_desk",
                    "x": 1.0,
                    "y": 1.0,
                    "rotation_deg": 90.0
                }
            ]
        }
        
        sanitized = sanitize_response(state, llm_response)
        
        # Trova il parent e il child sanificati
        san_parent = next(o for o in sanitized.objects if o.name == "parent_desk")
        san_child = next(o for o in sanitized.objects if o.name == "child_lamp")
        
        # Il parent deve essere nella posizione proposta
        self.assertAlmostEqual(san_parent.transform.location[0], 1.0, places=3)
        self.assertAlmostEqual(san_parent.transform.location[1], 1.0, places=3)
        self.assertAlmostEqual(san_parent.transform.rotation_euler[2], math.radians(90.0), places=3)
        
        # Il child deve essere ruotato solidalmente con il parent.
        # Poiche' il parent e' a (1,1) e ruotato di 90 gradi (verso +Y),
        # l'offset originale (+0.5 su X locale) ora punta verso +Y in coordinate mondo.
        # Quindi la posizione del child deve essere: (1.0, 1.5, 0.8)
        self.assertAlmostEqual(san_child.transform.location[0], 1.0, places=3)
        self.assertAlmostEqual(san_child.transform.location[1], 1.5, places=3)
        self.assertAlmostEqual(san_child.transform.rotation_euler[2], math.radians(90.0), places=3)


if __name__ == '__main__':
    unittest.main()
