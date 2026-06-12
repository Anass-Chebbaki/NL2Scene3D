# nl2scene3d/core/randomizer.py
"""
Disordinamento controllato del layout di una scena 3D.

Obiettivo:
    Disordinare la scena in modo plausibile: oggetti fuori posto ma non
    sovrapposti, non oltre i muri, con i valori Z originali intatti.

Regole di progetto:
    - La Z non viene MAI modificata. Un oggetto a 0.80 m (es. su una scrivania)
      resta a 0.80 m. Niente drop-to-floor: abbasserebbe mensole e monitor a terra.
    - I figli si muovono col padre tramite trasformazione rigida XY, mantenendo
      le posizioni relative invariate.
    - Ogni oggetto (o gruppo padre+figli) e' trattato come un blob con AABB
      espanso di collision_margin, cosi' visivamente non si sovrappongono mai.
    - Gli oggetti sono ordinati per volume decrescente: prima i pezzi grandi.

Modulo puro Python: nessuna dipendenza da bpy.
"""

from __future__ import annotations

import logging
import math
import random

from .geometry import collision_score, group_aabb_xy
from .models import RoomBounds, SceneObject, SceneState, Transform
from .settings import CONST, Constants

logger = logging.getLogger(__name__)


def _gather_descendants(
    root_name: str,
    by_name:   dict[str, SceneObject],
) -> list[SceneObject]:
    """
    Raccoglie ricorsivamente tutti i discendenti (figli, nipoti, ...) di un
    oggetto root, escluso il root stesso. Funzione pura a livello di modulo
    (niente closure su variabili di ciclo).
    """
    out: list[SceneObject] = []

    def rec(name: str) -> None:
        node = by_name.get(name)
        if node is None:
            return
        for cname in node.children:
            child = by_name.get(cname)
            if child is not None:
                out.append(child)
                rec(cname)

    rec(root_name)
    return out


# ---------------------------------------------------------------------------
# Trasformazione rigida di gruppo
# ---------------------------------------------------------------------------

def apply_rigid_transform(
    child:          SceneObject,
    old_parent_loc: list[float],
    old_parent_rz:  float,
    new_parent_loc: list[float],
    new_parent_rz:  float,
    original_z:     float | None = None,
) -> None:
    """
    Muove e ruota un figlio rigidamente rispetto al padre (solo piano XY).

    La Z non viene mai modificata: se `original_z` e' fornito viene usato,
    altrimenti la coordinata Z del figlio rimane invariata.
    Modifica child.transform in-place.

    Args:
        child:          Figlio da spostare.
        old_parent_loc: Posizione precedente del padre.
        old_parent_rz:  Rotazione Z precedente del padre.
        new_parent_loc: Nuova posizione del padre.
        new_parent_rz:  Nuova rotazione Z del padre.
        original_z:     Quota Z originale del figlio (se None, lascia invariata).
    """
    rel_x = child.transform.location[0] - old_parent_loc[0]
    rel_y = child.transform.location[1] - old_parent_loc[1]

    d_rz         = new_parent_rz - old_parent_rz
    cos_a, sin_a = math.cos(d_rz), math.sin(d_rz)

    child.transform.location[0] = new_parent_loc[0] + rel_x * cos_a - rel_y * sin_a
    child.transform.location[1] = new_parent_loc[1] + rel_x * sin_a + rel_y * cos_a

    if original_z is not None:
        child.transform.location[2] = original_z

    child.transform.rotation_euler[2] = (
        child.transform.rotation_euler[2] + d_rz
    ) % (2 * math.pi)


def _clamp_parent_group_location(
    orig_parent:   SceneObject,
    proposed_loc:  list[float],
    proposed_rz:   float,
    orig_children: list[SceneObject],
    room_bounds:   RoomBounds,
    wall_margin:   float,
) -> list[float]:
    """
    Vincola la posizione del padre cosi' che padre e tutti i figli restino
    completamente dentro i confini della stanza con il margine dato.

    Usa l'AABB combinato del gruppo per calcolare l'overflow su ogni lato
    e corregge la posizione del padre di conseguenza.
    """
    g_x_min, g_x_max, g_y_min, g_y_max = group_aabb_xy(
        orig_parent, proposed_loc, proposed_rz, orig_children, margin=0.0
    )

    px, py = proposed_loc[0], proposed_loc[1]

    overflow_left  = max(0.0, (room_bounds.x_min + wall_margin) - g_x_min)
    overflow_right = max(0.0, g_x_max - (room_bounds.x_max - wall_margin))
    overflow_front = max(0.0, (room_bounds.y_min + wall_margin) - g_y_min)
    overflow_back  = max(0.0, g_y_max - (room_bounds.y_max - wall_margin))

    dx = overflow_left  if overflow_left  > overflow_right else -overflow_right
    dy = overflow_front if overflow_front > overflow_back  else -overflow_back

    return [px + dx, py + dy, proposed_loc[2]]


# ---------------------------------------------------------------------------
# Generatori di posizione e rotazione
# ---------------------------------------------------------------------------

def _random_location(
    original_location: list[float],
    dimensions:        list[float],
    origin_offset:     list[float],
    rotation_z:        float,
    room_bounds:       RoomBounds,
    jitter_ratio:      float,
    wall_margin:       float,
    rng:               random.Random,
) -> list[float]:
    """
    Genera una posizione XY valida dentro la stanza.

    Tiene conto dell'AABB ruotato e dell'origin offset per garantire che
    l'oggetto rimanga completamente dentro i confini. La Z e' sempre
    quella originale.

    Il campionamento avviene in un'area centrata sulla posizione originale,
    con raggio pari a jitter_ratio * dimensione stanza (per un disordine
    localizzato ma plausibile). Se il range risultante e' invalido, viene
    usato l'intero spazio disponibile.
    """
    # Calcola l'offset dell'AABB rispetto all'origine (con la rotazione data).
    temp_tf = Transform(
        location=[0.0, 0.0, original_location[2]],
        rotation_euler=[0.0, 0.0, rotation_z],
        dimensions=dimensions,
        origin_offset=origin_offset,
    )
    t_xmin, t_xmax, t_ymin, t_ymax = temp_tf.aabb_xy(margin=0.0)

    # Range sicuro per l'origine (garantisce che l'intero AABB stia dentro la stanza).
    safe_x_min = room_bounds.x_min + wall_margin - t_xmin
    safe_x_max = room_bounds.x_max - wall_margin - t_xmax
    safe_y_min = room_bounds.y_min + wall_margin - t_ymin
    safe_y_max = room_bounds.y_max - wall_margin - t_ymax

    if safe_x_max <= safe_x_min or safe_y_max <= safe_y_min:
        # L'oggetto e' troppo grande per la stanza: lascialo dov'era.
        return list(original_location)

    # Restringe il range di campionamento attorno alla posizione originale.
    jitter_x = room_bounds.width * jitter_ratio
    jitter_y = room_bounds.depth * jitter_ratio
    cx_orig  = original_location[0]
    cy_orig  = original_location[1]

    range_x_min = max(safe_x_min, cx_orig - jitter_x)
    range_x_max = min(safe_x_max, cx_orig + jitter_x)
    range_y_min = max(safe_y_min, cy_orig - jitter_y)
    range_y_max = min(safe_y_max, cy_orig + jitter_y)

    # Fallback: se il range ristretto e' invalido, usa il range sicuro completo.
    if range_x_max < range_x_min:
        range_x_min, range_x_max = safe_x_min, safe_x_max
    if range_y_max < range_y_min:
        range_y_min, range_y_max = safe_y_min, safe_y_max

    new_x = rng.uniform(range_x_min, range_x_max)
    new_y = rng.uniform(range_y_min, range_y_max)

    return [new_x, new_y, original_location[2]]


def _random_rotation(original_rz: float, rng: random.Random) -> float:
    """Restituisce la rotazione Z originale + un multiplo casuale di 90 gradi."""
    delta = rng.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    return (original_rz + delta) % (2 * math.pi)


# ---------------------------------------------------------------------------
# SceneRandomizer
# ---------------------------------------------------------------------------

class SceneRandomizer:
    """
    Disordina artificialmente il layout di una scena 3D.

    Algoritmo:
        1. Raccoglie i root mobili (oggetti senza padre).
        2. Li ordina per volume decrescente (prima i pezzi grandi).
        3. Per ogni root prova fino a max_placement_attempts posizioni casuali:
            a. Genera una rotazione Z casuale (multiplo di 90 gradi).
            b. Genera una posizione valida coerente con quella rotazione e i confini.
            c. Verifica le collisioni (muri + mobili) con AABB espansi.
            d. Se non trova posizione libera, usa quella col punteggio piu' basso.
        4. Muove i figli con trasformazione rigida rispetto al padre.
    """

    def __init__(self, seed: int = 0, const: Constants = CONST) -> None:
        self.const = const
        self.seed  = seed
        self._rng  = random.Random(seed if seed != 0 else None)
        logger.info(
            "SceneRandomizer: seed=%s, jitter=%.2f, wall_margin=%.2f, collision_margin=%.2f.",
            seed, const.jitter_ratio, const.wall_margin, const.collision_margin,
        )

    def randomize(self, state: SceneState) -> SceneState:
        """
        Applica la randomizzazione a una copia profonda dello SceneState.

        L'originale non viene mai modificato.
        """
        if state.room_bounds is None:
            raise ValueError("SceneState senza room_bounds. Estrai prima la scena.")

        bounds = state.room_bounds
        logger.info(
            "Randomizzo '%s': %d oggetti mobili (%d root).",
            state.scene_name, len(state.movable_objects), len(state.root_movable_objects),
        )

        new_objects: list[SceneObject] = [obj.copy() for obj in state.objects]
        by_name = {obj.name: obj for obj in new_objects}

        # Gli oggetti fissi sono gia' "piazzati" come ostacoli.
        placed: list[SceneObject] = [obj for obj in new_objects if not obj.is_movable]

        # Root mobili ordinati per volume decrescente.
        roots = sorted(
            [obj for obj in new_objects if obj.is_movable and obj.is_root],
            key=lambda o: (
                o.transform.dimensions[0]
                * o.transform.dimensions[1]
                * o.transform.dimensions[2]
            ),
            reverse=True,
        )

        placed_count  = 0
        fallback_count = 0

        for root in roots:
            old_loc = list(root.transform.location)
            old_rz  = root.transform.rotation_euler[2]

            # Valuta la posizione corrente come candidato iniziale.
            best_obj   = root.copy()
            best_score = self._group_collision_score(best_obj, root, by_name, placed, bounds)

            # Prova posizioni casuali e tieni quella col punteggio migliore.
            for _ in range(self.const.max_placement_attempts):
                candidate = root.copy()

                new_rz = _random_rotation(old_rz, self._rng)
                candidate.transform.rotation_euler[2] = new_rz

                new_loc = _random_location(
                    old_loc,
                    candidate.transform.dimensions,
                    candidate.transform.origin_offset,
                    new_rz,
                    bounds,
                    self.const.jitter_ratio,
                    self.const.wall_margin,
                    self._rng,
                )
                candidate.transform.location = new_loc

                score = self._group_collision_score(candidate, root, by_name, placed, bounds)
                if score == 0.0:
                    best_obj, best_score = candidate, 0.0
                    break
                if score < best_score:
                    best_score, best_obj = score, candidate

            if best_score > 0.0:
                fallback_count += 1
                logger.debug(
                    "'%s': nessuna posizione libera in %d tentativi (fallback score=%.3f).",
                    root.name, self.const.max_placement_attempts, best_score,
                )

            # Applica la posizione migliore trovata al root nella lista di lavoro.
            final_root = by_name[root.name]
            new_loc    = list(best_obj.transform.location)
            new_rz     = best_obj.transform.rotation_euler[2]

            # Raccoglie i figli originali per il clamp di gruppo.
            orig_by_name = {o.name: o for o in state.objects}
            orig_children = _gather_descendants(root.name, orig_by_name)

            # Vincola il gruppo dentro i confini della stanza.
            clamped_loc = _clamp_parent_group_location(
                root, new_loc, new_rz, orig_children, bounds, self.const.wall_margin
            )
            if clamped_loc != new_loc:
                new_loc = clamped_loc

            # Blocco assoluto della Z: mai modificata.
            new_loc[2] = root.transform.location[2]

            final_root.transform.location         = new_loc
            final_root.transform.rotation_euler[2] = new_rz
            placed.append(final_root)
            placed_count += 1

            # Sposta i figli con trasformazione rigida.
            self._move_children(
                root_name=root.name,
                old_loc=old_loc,
                old_rz=old_rz,
                new_loc=new_loc,
                new_rz=new_rz,
                by_name=by_name,
                placed=placed,
                orig_by_name=orig_by_name,
            )

        logger.info(
            "Randomizzazione completa: %d root piazzati, %d con fallback.",
            placed_count, fallback_count,
        )

        return SceneState(
            scene_name=state.scene_name,
            objects=new_objects,
            room_bounds=bounds,
            pipeline_step="randomized",
            metadata={
                "randomizer_seed": self.seed,
                "placed_count":    placed_count,
                "fallback_count":  fallback_count,
            },
        )

    def _group_collision_score(
        self,
        root_candidate:    SceneObject,
        original_root:     SceneObject,
        all_objects_by_name: dict[str, SceneObject],
        placed_objects:    list[SceneObject],
        bounds:            RoomBounds,
    ) -> float:
        """
        Calcola il punteggio di collisione cumulativo del gruppo (padre + discendenti)
        alla posizione candidata.

        Valuta solo le collisioni spaziali; il check sulla Z non viene eseguito
        perche' la coordinata Z e' sempre bloccata al valore originale.
        """
        total_score = collision_score(
            root_candidate, placed_objects,
            wall_margin=self.const.wall_margin,
            furniture_margin=self.const.collision_margin,
            room_bounds=bounds,
        )

        def _collect_and_score_children(
            parent_name:        str,
            current_parent_loc: list[float],
            current_parent_rz:  float,
        ):
            """Valuta ricorsivamente i figli, trasformati rigidamente rispetto al padre candidato."""
            nonlocal total_score
            orig_parent = all_objects_by_name[parent_name]

            for child_name in orig_parent.children:
                orig_child  = all_objects_by_name[child_name]
                moved_child = orig_child.copy()

                apply_rigid_transform(
                    moved_child,
                    old_parent_loc=orig_parent.transform.location,
                    old_parent_rz=orig_parent.transform.rotation_euler[2],
                    new_parent_loc=current_parent_loc,
                    new_parent_rz=current_parent_rz,
                    original_z=orig_child.transform.location[2],
                )

                total_score += collision_score(
                    moved_child, placed_objects,
                    wall_margin=self.const.wall_margin,
                    furniture_margin=self.const.collision_margin,
                    room_bounds=bounds,
                )

                # Penalita' aggiuntiva se il figlio e' fuori dai confini.
                if bounds is not None and not bounds.contains_aabb(
                    moved_child.transform.aabb_xy(margin=0.0),
                    margin=self.const.wall_margin,
                ):
                    total_score += 100.0

                _collect_and_score_children(
                    child_name,
                    moved_child.transform.location,
                    moved_child.transform.rotation_euler[2],
                )

        _collect_and_score_children(
            original_root.name,
            root_candidate.transform.location,
            root_candidate.transform.rotation_euler[2],
        )

        return total_score

    def _move_children(
        self,
        root_name:    str,
        old_loc:      list[float],
        old_rz:       float,
        new_loc:      list[float],
        new_rz:       float,
        by_name:      dict[str, SceneObject],
        placed:       list[SceneObject],
        orig_by_name: dict[str, SceneObject] | None = None,
    ) -> None:
        """
        Muove ricorsivamente tutti i discendenti del root con trasformazione rigida XY.

        Per ogni figlio recupera la Z originale dallo snapshot orig_by_name
        (se disponibile) per garantire che la quota non venga mai alterata.
        """
        root_obj = by_name.get(root_name)
        if root_obj is None:
            return

        for child_name in root_obj.children:
            child = by_name.get(child_name)
            if child is None:
                continue

            original_child_z = None
            orig_child_loc   = old_loc
            orig_child_rz    = old_rz

            if orig_by_name:
                orig_child = orig_by_name.get(child_name)
                if orig_child:
                    original_child_z = orig_child.transform.location[2]
                    orig_child_loc   = list(orig_child.transform.location)
                    orig_child_rz    = orig_child.transform.rotation_euler[2]

            apply_rigid_transform(
                child, old_loc, old_rz, new_loc, new_rz,
                original_z=original_child_z,
            )
            placed.append(child)

            # Ricorsione sui figli del figlio.
            self._move_children(
                child_name, orig_child_loc, orig_child_rz,
                list(child.transform.location), child.transform.rotation_euler[2],
                by_name, placed, orig_by_name,
            )