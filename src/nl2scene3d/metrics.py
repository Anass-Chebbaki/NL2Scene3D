"""
Calcolo delle metriche di valutazione qualitativa della pipeline.

Questo modulo calcola indicatori numerici per misurare quanto
la pipeline abbia migliorato il layout della scena rispetto
allo stato disordinato e confrontandolo con l'originale.

Le metriche calcolate sono:
- Distanza media dalla posizione originale (per stato disordinato e riordinato)
- Variazione angolare media sull'asse Z
- Score di miglioramento: quanto il riordinato si avvicina all'originale
  rispetto al disordinato (0.0 = nessun miglioramento, 1.0 = originale esatto)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from nl2scene3d.models import SceneState

logger = logging.getLogger(__name__)

@dataclass
class SceneMetrics:
    """
    Metriche calcolate per uno specifico stato della scena.

    Attributes:
        scene_name: Nome della scena di riferimento.
        pipeline_step: Step della pipeline a cui si riferiscono le metriche.
        mean_position_delta_meters: Distanza media (in metri) tra le posizioni
            degli oggetti in questo stato e quelle dello stato originale.
            Calcolata solo per gli oggetti movibili.
        mean_rotation_delta_radians: Differenza angolare media (in radianti)
            sull'asse Z tra questo stato e lo stato originale.
        object_count_movable: Numero di oggetti movibili considerati.
        improvement_score: Score in [0.0, 1.0] che indica quanto il layout
            si avvicina all'originale rispetto al disordinato.
            0.0 = uguale al disordinato, 1.0 = identico all'originale.
            None se non calcolabile (manca il riferimento disordinato).
    """

    scene_name: str
    pipeline_step: str
    mean_position_delta_meters: float
    mean_rotation_delta_radians: float
    object_count_movable: int
    improvement_score: float | None = None
    per_object_details: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializza le metriche in dizionario compatibile con JSON."""
        return {
            "scene_name": self.scene_name,
            "pipeline_step": self.pipeline_step,
            "mean_position_delta_meters": round(self.mean_position_delta_meters, 4),
            "mean_rotation_delta_radians": round(self.mean_rotation_delta_radians, 4),
            "object_count_movable": self.object_count_movable,
            "improvement_score": (
                round(self.improvement_score, 4)
                if self.improvement_score is not None
                else None
            ),
            "per_object_details": self.per_object_details,
        }

    def summary_line(self) -> str:
        """
        Restituisce una stringa riassuntiva leggibile delle metriche.

        Returns:
            Stringa formattata con le metriche principali.
        """
        improvement_str = (
            f"{self.improvement_score:.3f}"
            if self.improvement_score is not None
            else "N/A"
        )
        return (
            f"[{self.pipeline_step}] "
            f"delta_pos={self.mean_position_delta_meters:.3f}m, "
            f"delta_rot={math.degrees(self.mean_rotation_delta_radians):.1f}deg, "
            f"improvement={improvement_str}"
        )

def _euclidean_distance_2d(
    loc_a: list[float],
    loc_b: list[float],
) -> float:
    """
    Calcola la distanza euclidea 2D tra due posizioni (piano XY).

    La coordinata Z viene ignorata per confrontare solo lo spostamento
    planimetrico degli oggetti.

    Args:
        loc_a: Prima posizione [x, y, z].
        loc_b: Seconda posizione [x, y, z].

    Returns:
        Distanza euclidea nel piano XY in metri.
    """
    dx = loc_a[0] - loc_b[0]
    dy = loc_a[1] - loc_b[1]
    return math.sqrt(dx * dx + dy * dy)

def _angular_difference_z(
    rot_a: list[float],
    rot_b: list[float],
) -> float:
    """
    Calcola la differenza angolare minima sull'asse Z tra due rotazioni.

    Usa la differenza angolare normalizzata in [0, pi] per evitare
    ambiguita' con angoli equivalenti (es. 0 e 2*pi sono identici).

    Args:
        rot_a: Prima rotazione [rx, ry, rz] in radianti.
        rot_b: Seconda rotazione [rx, ry, rz] in radianti.

    Returns:
        Differenza angolare sull'asse Z in [0, pi] radianti.
    """
    diff = abs(rot_a[2] - rot_b[2])
    # Normalizza in [0, pi]: angoli supplementari sono equivalenti
    # per oggetti simmetrici, ma usiamo la differenza assoluta minima
    diff = diff % (2.0 * math.pi)
    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return diff

def compute_metrics(
    reference_state: SceneState,
    evaluated_state: SceneState,
    disordered_state: SceneState | None = None,
) -> SceneMetrics:
    """
    Calcola le metriche di qualita' per uno stato della scena.

    Confronta lo stato valutato con lo stato di riferimento (originale)
    per determinare quanto il riordino si avvicini alla configurazione
    professionale originale.

    Args:
        reference_state: Stato originale della scena (ground truth).
        evaluated_state: Stato da valutare (riordinato o raffinato).
        disordered_state: Stato disordinato (opzionale, usato per calcolare
                          l'improvement_score).

    Returns:
        SceneMetrics con le metriche calcolate.
    """
    position_deltas: list[float] = []
    rotation_deltas: list[float] = []
    disordered_deltas: list[float] = []
    per_object: dict[str, dict] = {}

    for ref_obj in reference_state.movable_objects:
        eval_obj = evaluated_state.get_object_by_name(ref_obj.name)
        if eval_obj is None or not eval_obj.is_movable:
            continue

        pos_delta = _euclidean_distance_2d(
            ref_obj.transform.location,
            eval_obj.transform.location,
        )
        rot_delta = _angular_difference_z(
            ref_obj.transform.rotation_euler,
            eval_obj.transform.rotation_euler,
        )

        position_deltas.append(pos_delta)
        rotation_deltas.append(rot_delta)

        obj_details: dict = {
            "position_delta_meters": round(pos_delta, 4),
            "rotation_delta_degrees": round(math.degrees(rot_delta), 2),
        }

        # Calcola il delta dal disordinato se disponibile
        if disordered_state is not None:
            dis_obj = disordered_state.get_object_by_name(ref_obj.name)
            if dis_obj is not None:
                dis_delta = _euclidean_distance_2d(
                    ref_obj.transform.location,
                    dis_obj.transform.location,
                )
                disordered_deltas.append(dis_delta)
                obj_details["disordered_delta_meters"] = round(dis_delta, 4)

        per_object[ref_obj.name] = obj_details

    if not position_deltas:
        logger.warning(
            "Nessun oggetto movibile in comune tra lo stato di riferimento "
            "e quello valutato. Metriche non calcolabili."
        )
        return SceneMetrics(
            scene_name=evaluated_state.scene_name,
            pipeline_step=evaluated_state.pipeline_step,
            mean_position_delta_meters=0.0,
            mean_rotation_delta_radians=0.0,
            object_count_movable=0,
        )

    mean_pos_delta = sum(position_deltas) / len(position_deltas)
    mean_rot_delta = sum(rotation_deltas) / len(rotation_deltas)

    # Calcolo improvement_score: quanto il riordinato si avvicina all'originale
    improvement_score: float | None = None
    if disordered_deltas and len(disordered_deltas) == len(position_deltas):
        mean_dis_delta = sum(disordered_deltas) / len(disordered_deltas)
        if mean_dis_delta > 0.0:
            # Score: 1.0 se uguale all'originale, 0.0 se uguale al disordinato
            improvement_score = 1.0 - (mean_pos_delta / mean_dis_delta)
            # Clamp in [0.0, 1.0] (puo' essere negativo se peggioriamo)
            improvement_score = max(0.0, min(1.0, improvement_score))

    metrics = SceneMetrics(
        scene_name=evaluated_state.scene_name,
        pipeline_step=evaluated_state.pipeline_step,
        mean_position_delta_meters=mean_pos_delta,
        mean_rotation_delta_radians=mean_rot_delta,
        object_count_movable=len(position_deltas),
        improvement_score=improvement_score,
        per_object_details=per_object,
    )

    logger.info(
        "Metriche calcolate per '%s': %s",
        evaluated_state.scene_name,
        metrics.summary_line(),
    )
    return metrics

def compute_pipeline_metrics(
    original_state: SceneState,
    randomized_state: SceneState,
    reordered_state: SceneState,
    refined_state: SceneState,
) -> dict[str, SceneMetrics]:
    """
    Calcola le metriche per tutti gli step della pipeline in una sola chiamata.

    Args:
        original_state: Stato originale (ground truth).
        randomized_state: Stato dopo la randomizzazione.
        reordered_state: Stato dopo il riordino LLM.
        refined_state: Stato dopo il refinement visivo.

    Returns:
        Dizionario step_name -> SceneMetrics.
    """
    results: dict[str, SceneMetrics] = {}

    results["randomized"] = compute_metrics(
        reference_state=original_state,
        evaluated_state=randomized_state,
    )

    results["reordered"] = compute_metrics(
        reference_state=original_state,
        evaluated_state=reordered_state,
        disordered_state=randomized_state,
    )

    results["refined"] = compute_metrics(
        reference_state=original_state,
        evaluated_state=refined_state,
        disordered_state=randomized_state,
    )

    logger.info(
        "Metriche pipeline complete per scena '%s':",
        original_state.scene_name,
    )
    for step, m in results.items():
        logger.info("  %s", m.summary_line())

    return results
