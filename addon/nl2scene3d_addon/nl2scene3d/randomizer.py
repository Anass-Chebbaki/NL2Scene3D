# nl2scene3d/randomizer.py
"""
Randomizzazione controllata del layout di una scena 3D.

Obiettivo: disorganizzare la scena in modo "plausibile" — oggetti fuori posto
ma non sovrapposti, non fuori dai muri, con le Z originali intatte.

Principi fondamentali:
- La coordinata Z non viene MAI modificata. Un oggetto che nell'originale era
  a 0.80m (sul tavolo) rimane a 0.80m anche dopo la randomizzazione.
  La simulazione fisica (drop to floor) è esclusa per design: causerebbe
  mensole sul pavimento, monitor sul pavimento, ecc.
- I figli (oggetti appoggiati/dentro un parent) si muovono insieme al parent
  con una trasformazione rigida, mantenendo la loro posizione relativa.
- Ogni oggetto (o gruppo parent+figli) viene trattato come un blob con AABB
  espansa del collision_margin, così anche in caso di "collisione accettata"
  gli oggetti non sono mai visivamente intrecciati.
- Gli oggetti vengono ordinati per volume decrescente: i pezzi grandi
  (letti, armadi) vengono piazzati prima, poi i più piccoli si incastrano.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Optional

from nl2scene3d.config import RandomizerConfig
from nl2scene3d.models import RoomBounds, SceneObject, SceneState, Transform
from nl2scene3d.utils.geometry import collision_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trasformazione rigida del gruppo
# ---------------------------------------------------------------------------

def apply_rigid_transform(
    child: SceneObject,
    old_parent_loc: list[float],
    old_parent_rz: float,
    new_parent_loc: list[float],
    new_parent_rz: float,
) -> None:
    """
    Sposta e ruota un oggetto figlio in modo rigido rispetto al suo parent.

    Calcola la posizione relativa del figlio rispetto al vecchio parent,
    la ruota del delta angolare, e la riapplica al nuovo parent.
    La Z del figlio viene aggiornata solo del delta Z del parent (traslazione pura).

    Modifica child.transform in-place.
    """
    # Vettore relativo figlio → old parent nel piano XY
    rel_x = child.transform.location[0] - old_parent_loc[0]
    rel_y = child.transform.location[1] - old_parent_loc[1]
    rel_z = child.transform.location[2] - old_parent_loc[2]

    # Delta rotazione Z del parent
    d_rz = new_parent_rz - old_parent_rz
    cos_a, sin_a = math.cos(d_rz), math.sin(d_rz)

    # Ruota il vettore relativo attorno all'asse Z
    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a

    # Nuova posizione assoluta del figlio
    child.transform.location[0] = new_parent_loc[0] + rot_x
    child.transform.location[1] = new_parent_loc[1] + rot_y
    child.transform.location[2] = new_parent_loc[2] + rel_z  # Z: solo traslazione, no rotazione

    # Ruota anche il figlio di d_rz sull'asse Z
    child.transform.rotation_euler[2] = (
        child.transform.rotation_euler[2] + d_rz
    ) % (2 * math.pi)


def _group_aabb_xy(
    orig_parent: SceneObject,
    proposed_loc: list[float],
    proposed_rz: float,
    orig_children: list[SceneObject],
    margin: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Calcola l'AABB 2D combinata (XY) di un gruppo parent+figli nella posizione/rotazione proposta.
    Usa l'AABB reale di ogni membro (con la rotazione corretta) tramite la classe Transform,
    garantendo che venga sempre tenuto conto di origin_offset e rotazione Z.
    """
    old_parent_loc = orig_parent.transform.location
    old_parent_rz = orig_parent.transform.rotation_euler[2]
    d_rz = proposed_rz - old_parent_rz
    cos_a, sin_a = math.cos(d_rz), math.sin(d_rz)

    # Transform temporanea del parent
    temp_parent_tf = Transform(
        location=[proposed_loc[0], proposed_loc[1], orig_parent.transform.location[2]],
        rotation_euler=[
            orig_parent.transform.rotation_euler[0],
            orig_parent.transform.rotation_euler[1],
            proposed_rz
        ],
        dimensions=orig_parent.transform.dimensions,
        origin_offset=orig_parent.transform.origin_offset
    )
    x_min, x_max, y_min, y_max = temp_parent_tf.aabb_xy(margin=margin)

    # Estendi l'AABB con ogni figlio nella sua nuova posizione rigida
    for orig_child in orig_children:
        rel_x = orig_child.transform.location[0] - old_parent_loc[0]
        rel_y = orig_child.transform.location[1] - old_parent_loc[1]
        
        # Ruota il vettore relativo
        new_cx = proposed_loc[0] + rel_x * cos_a - rel_y * sin_a
        new_cy = proposed_loc[1] + rel_x * sin_a + rel_y * cos_a
        c_rz = (orig_child.transform.rotation_euler[2] + d_rz) % (2 * math.pi)

        temp_child_tf = Transform(
            location=[new_cx, new_cy, orig_child.transform.location[2]],
            rotation_euler=[
                orig_child.transform.rotation_euler[0],
                orig_child.transform.rotation_euler[1],
                c_rz
            ],
            dimensions=orig_child.transform.dimensions,
            origin_offset=orig_child.transform.origin_offset
        )
        cx_min, cx_max, cy_min, cy_max = temp_child_tf.aabb_xy(margin=margin)
        x_min = min(x_min, cx_min)
        x_max = max(x_max, cx_max)
        y_min = min(y_min, cy_min)
        y_max = max(y_max, cy_max)

    return x_min, x_max, y_min, y_max


def _clamp_parent_group_location_rand(
    orig_parent: SceneObject,
    proposed_loc: list[float],
    proposed_rz: float,
    orig_children: list[SceneObject],
    room_bounds,
    wall_margin: float,
) -> list[float]:
    """
    Clampa la posizione del parent in modo che sia il parent stesso sia tutti i suoi figli
    rimangano all'interno dei bounds della stanza, rispettando il wall_margin.
    Usa l'AABB combinata del gruppo per evitare errori di doppia rotazione.
    """
    # Calcola l'AABB reale del gruppo nella posizione proposta
    g_x_min, g_x_max, g_y_min, g_y_max = _group_aabb_xy(
        orig_parent, proposed_loc, proposed_rz, orig_children, margin=0.0
    )

    # Calcola i delta di eccesso per ogni lato
    px, py = proposed_loc[0], proposed_loc[1]

    # Sposta il centro del parent di quanto è necessario per rientrare nei bounds
    overflow_left  = max(0.0, (room_bounds.x_min + wall_margin) - g_x_min)
    overflow_right = max(0.0, g_x_max - (room_bounds.x_max - wall_margin))
    overflow_front = max(0.0, (room_bounds.y_min + wall_margin) - g_y_min)
    overflow_back  = max(0.0, g_y_max - (room_bounds.y_max - wall_margin))

    # Non applicare entrambi i lati contemporaneamente: priorità al lato più grande
    dx = overflow_left if overflow_left > overflow_right else -overflow_right
    dy = overflow_front if overflow_front > overflow_back else -overflow_back

    return [px + dx, py + dy, proposed_loc[2]]


# ---------------------------------------------------------------------------
# Posizione casuale per un oggetto
# ---------------------------------------------------------------------------

def _random_location(
    original_location: list[float],
    dimensions: list[float],
    origin_offset: list[float],
    rotation_z: float,
    room_bounds: RoomBounds,
    jitter_ratio: float,
    wall_margin: float,
    rng: random.Random,
) -> list[float]:
    """Genera una posizione XY all'interno dei bounds con jittering tenendo conto dell'origin_offset."""
    # Creiamo una Transform fittizia all'origine [0.0, 0.0] con la rotazione proposta
    # per calcolare l'esatta estensione dell'AABB ruotata tenendo conto dell'offset origine.
    temp_tf = Transform(
        location=[0.0, 0.0, original_location[2]],
        rotation_euler=[0.0, 0.0, rotation_z],
        dimensions=dimensions,
        origin_offset=origin_offset
    )
    t_xmin, t_xmax, t_ymin, t_ymax = temp_tf.aabb_xy(margin=0.0)

    # Range fisicamente valido (il centro origine deve essere posizionato in modo che l'AABB stia dentro i muri)
    safe_x_min = room_bounds.x_min + wall_margin - t_xmin
    safe_x_max = room_bounds.x_max - wall_margin - t_xmax
    safe_y_min = room_bounds.y_min + wall_margin - t_ymin
    safe_y_max = room_bounds.y_max - wall_margin - t_ymax

    if safe_x_max <= safe_x_min or safe_y_max <= safe_y_min:
        # Oggetto troppo grande per la stanza — lascialo dov'è
        return list(original_location)

    # Restringe il range al jitter attorno alla posizione originale
    jitter_x = room_bounds.width * jitter_ratio
    jitter_y = room_bounds.depth * jitter_ratio

    cx_orig = original_location[0]
    cy_orig = original_location[1]

    range_x_min = max(safe_x_min, cx_orig - jitter_x)
    range_x_max = min(safe_x_max, cx_orig + jitter_x)
    range_y_min = max(safe_y_min, cy_orig - jitter_y)
    range_y_max = min(safe_y_max, cy_orig + jitter_y)

    # Se il jitter ha ristretto il range fino a collasso, usa il safe range completo
    if range_x_max < range_x_min:
        range_x_min, range_x_max = safe_x_min, safe_x_max
    if range_y_max < range_y_min:
        range_y_min, range_y_max = safe_y_min, safe_y_max

    new_x = rng.uniform(range_x_min, range_x_max)
    new_y = rng.uniform(range_y_min, range_y_max)

    return [new_x, new_y, original_location[2]]  # Z invariata


def _random_rotation(original_rz: float, rng: random.Random) -> float:
    """Ruota di un multiplo casuale di 90°. Solo asse Z."""
    delta = rng.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    return (original_rz + delta) % (2 * math.pi)


# ---------------------------------------------------------------------------
# Randomizer principale
# ---------------------------------------------------------------------------

class SceneRandomizer:
    """
    Disorganizza artificialmente il layout di una scena 3D.

    Funzionamento:
    1. Raccoglie tutti gli oggetti root movibili (quelli senza parent).
    2. Li ordina per volume decrescente (i grandi vanno piazzati prima).
    3. Per ogni root, tenta max_placement_attempts posizioni casuali.
       - Ad ogni tentativo genera prima la rotazione (multiplo di 90°),
         poi la posizione compatibile con quella rotazione e i room bounds.
       - Verifica collisioni con muri e mobili già piazzati usando AABB espansi.
       - Se nessuna posizione è libera, usa quella con il punteggio di collisione
         più basso (collision_score) invece di lasciare l'oggetto dov'era.
    4. Sposta i figli con trasformazione rigida rispetto al parent.
    """

    def __init__(self, config: Optional[RandomizerConfig] = None) -> None:
        self.config = config or RandomizerConfig()
        seed = self.config.seed if self.config.seed != 0 else None
        self._rng = random.Random(seed)
        logger.info(
            "SceneRandomizer inizializzato. seed=%s, jitter=%.2f, wall_margin=%.2f, collision_margin=%.2f.",
            seed, self.config.jitter_ratio, self.config.wall_margin, self.config.collision_margin,
        )

    def randomize(self, state: SceneState) -> SceneState:
        """
        Applica la randomizzazione a una copia della scena.

        Non modifica lo stato originale.

        Args:
            state: SceneState con pipeline_step='original' e grouping pre-calcolato.

        Returns:
            Nuovo SceneState con pipeline_step='randomized'.
        """
        if state.room_bounds is None:
            raise ValueError("SceneState senza room_bounds. Estrai prima la scena con SceneLoader.")

        bounds = state.room_bounds
        logger.info(
            "Randomizzazione '%s': %d oggetti movibili (%d root).",
            state.scene_name,
            len(state.movable_objects),
            len(state.root_movable_objects),
        )

        # Copia profonda di tutti gli oggetti
        new_objects: list[SceneObject] = [obj.copy() for obj in state.objects]
        by_name = {obj.name: obj for obj in new_objects}

        # Già piazzati (strutturali) — partono come "ostacoli fissi"
        placed: list[SceneObject] = [obj for obj in new_objects if obj.category == "structural"]

        # Root movibili ordinati per volume decrescente
        roots = sorted(
            [obj for obj in new_objects if obj.is_movable and obj.is_root],
            key=lambda o: o.transform.dimensions[0]
                * o.transform.dimensions[1]
                * o.transform.dimensions[2],
            reverse=True,
        )

        placed_count = 0
        fallback_count = 0

        for root in roots:
            old_loc = list(root.transform.location)
            old_rz = root.transform.rotation_euler[2]

            best_obj: Optional[SceneObject] = None
            best_score: float = float("inf")

            for attempt in range(self.config.max_placement_attempts):
                candidate = root.copy()

                # 1. Genera la rotazione
                new_rz = _random_rotation(old_rz, self._rng)
                candidate.transform.rotation_euler[2] = new_rz

                # 2. Genera la posizione coerente con la rotazione
                new_loc = _random_location(
                    old_loc,
                    candidate.transform.dimensions,
                    candidate.transform.origin_offset,
                    new_rz,
                    bounds,
                    self.config.jitter_ratio,
                    self.config.wall_margin,
                    self._rng,
                )
                candidate.transform.location = new_loc

                # 3. Verifica collisioni sull'intero gruppo (root + figli)
                score = self._group_collision_score(
                    root_candidate=candidate,
                    original_root=root,
                    all_objects_by_name=by_name,
                    placed_objects=placed,
                    bounds=bounds,
                )

                if score == 0.0:
                    # Posizione perfettamente libera per tutto il gruppo
                    best_obj = candidate
                    break

                # Tiene traccia della posizione meno problematica come fallback
                if score < best_score:
                    best_score = score
                    best_obj = candidate

            # best_obj è garantito essere non-None (almeno un tentativo viene fatto)
            assert best_obj is not None

            if best_score > 0.0:
                fallback_count += 1
                logger.debug(
                    "'%s': nessuna posizione libera dopo %d tentativi. "
                    "Usata la migliore (score=%.3f).",
                    root.name, self.config.max_placement_attempts, best_score,
                )

            # Applica la posizione scelta al root nella lista new_objects
            final_root = by_name[root.name]
            new_loc = best_obj.transform.location
            new_rz = best_obj.transform.rotation_euler[2]

            # Clampa l'intera posizione finale per garantire che tutto il gruppo sia al sicuro entro i muri
            if bounds is not None:
                orig_children = []
                orig_by_name = {o.name: o for o in state.objects}
                def gather_children(p_name: str):
                    p_obj = orig_by_name.get(p_name)
                    if p_obj:
                        for c_name in p_obj.children:
                            c_obj = orig_by_name.get(c_name)
                            if c_obj:
                                orig_children.append(c_obj)
                                gather_children(c_name)
                gather_children(root.name)

                clamped_loc = _clamp_parent_group_location_rand(
                    root,
                    new_loc,
                    new_rz,
                    orig_children,
                    bounds,
                    self.config.wall_margin,
                )
                if clamped_loc != new_loc:
                    logger.debug("Randomizer group clamp per '%s': %s -> %s", root.name, new_loc, clamped_loc)
                    new_loc = clamped_loc

            final_root.transform.location = new_loc
            final_root.transform.rotation_euler[2] = new_rz

            placed.append(final_root)
            placed_count += 1

            # 4. Sposta i figli con trasformazione rigida
            self._move_children(
                root_name=root.name,
                old_loc=old_loc,
                old_rz=old_rz,
                new_loc=new_loc,
                new_rz=new_rz,
                by_name=by_name,
                placed=placed,
                bounds=bounds,
            )

        logger.info(
            "Randomizzazione completata: %d root spostati, %d con fallback posizione.",
            placed_count, fallback_count,
        )

        return SceneState(
            scene_name=state.scene_name,
            objects=new_objects,
            room_bounds=bounds,
            pipeline_step="randomized",
            metadata={
                "randomizer_seed": self.config.seed,
                "placed_count": placed_count,
                "fallback_count": fallback_count,
            },
        )

    def _group_collision_score(
        self,
        root_candidate: SceneObject,
        original_root: SceneObject,
        all_objects_by_name: dict[str, SceneObject],
        placed_objects: list[SceneObject],
        bounds,
    ) -> float:
        """
        Calcola il punteggio di collisione cumulativo per l'intero gruppo
        spostato nella nuova posizione del root.
        """
        # 1. Punteggio del root stesso (incluso check contenimento in bounds)
        total_score = collision_score(
            root_candidate,
            placed_objects,
            wall_margin=self.config.wall_margin,
            furniture_margin=self.config.collision_margin,
            room_bounds=bounds,
        )

        # 2. Punteggio per ogni figlio (spostato rigidamente rispetto al root_candidate)
        def _collect_and_score_children(parent_name: str, current_parent_loc: list[float], current_parent_rz: float):
            nonlocal total_score
            orig_parent = all_objects_by_name[parent_name]
            for child_name in orig_parent.children:
                orig_child = all_objects_by_name[child_name]
                # Crea una copia temporanea del figlio spostata
                moved_child = orig_child.copy()
                apply_rigid_transform(
                    moved_child,
                    old_parent_loc=orig_parent.transform.location,
                    old_parent_rz=orig_parent.transform.rotation_euler[2],
                    new_parent_loc=current_parent_loc,
                    new_parent_rz=current_parent_rz,
                )

                # 1. Collisione per questo figlio contro altri oggetti (incluso check bounds)
                total_score += collision_score(
                    moved_child,
                    placed_objects,
                    wall_margin=self.config.wall_margin,
                    furniture_margin=self.config.collision_margin,
                    room_bounds=bounds,
                )

                # 2. Verifica bounds per questo figlio (se fuori o troppo vicino ai muri, penalità pesante)
                if not bounds.contains_aabb(moved_child.transform.aabb_xy(margin=0.0), margin=self.config.wall_margin):
                    total_score += 100.0  # Penalità bloccante: il gruppo non può uscire dai muri

                # Ricorsione per i figli dei figli
                _collect_and_score_children(
                    child_name,
                    moved_child.transform.location,
                    moved_child.transform.rotation_euler[2]
                )

        _collect_and_score_children(
            original_root.name,
            root_candidate.transform.location,
            root_candidate.transform.rotation_euler[2]
        )

        return total_score

    def _move_children(
        self,
        root_name: str,
        old_loc: list[float],
        old_rz: float,
        new_loc: list[float],
        new_rz: float,
        by_name: dict[str, SceneObject],
        placed: list[SceneObject],
        bounds,
    ) -> None:
        """
        Sposta ricorsivamente tutti i discendenti del root con trasformazione rigida.
        """
        root_obj = by_name.get(root_name)
        if root_obj is None:
            return

        for child_name in root_obj.children:
            child = by_name.get(child_name)
            if child is None:
                continue

            old_child_loc = list(child.transform.location)
            old_child_rz = child.transform.rotation_euler[2]

            apply_rigid_transform(child, old_loc, old_rz, new_loc, new_rz)
            placed.append(child)

            # Ricorsione per i nipoti
            self._move_children(
                child_name, old_child_loc, old_child_rz,
                child.transform.location, child.transform.rotation_euler[2],
                by_name, placed, bounds,
            )