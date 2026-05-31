# nl2scene3d/metrics.py
"""
Qualitative evaluation metrics for the NL2Scene3D pipeline.

Computes numerical indicators that measure how well a reorganized layout
recovers the original scene configuration compared to the disordered state.

Metrics computed:
  - Mean XY position delta from the original (meters)
  - Mean Z-axis rotation delta from the original (radians)
  - Improvement score in [0.0, 1.0]:
      0.0 = no improvement over the disordered state
      1.0 = identical to the original
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from nl2scene3d.models import SceneState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SceneMetrics:
    """
    Metrics for a specific scene state in the pipeline.

    Attributes:
        scene_name:                  Name of the reference scene.
        pipeline_step:               Step label these metrics refer to.
        mean_position_delta_meters:  Mean XY distance between movable objects
                                     in this state and the original.
        mean_rotation_delta_radians: Mean Z-axis rotation difference from
                                     the original (radians).
        object_count_movable:        Number of movable objects included in
                                     the calculation.
        improvement_score:           Value in [0.0, 1.0] indicating how much
                                     the layout recovered toward the original
                                     compared to the disordered state.
                                     None if the disordered state was not provided.
        per_object_details:          Per-object metric breakdown.
    """

    scene_name:                  str
    pipeline_step:               str
    mean_position_delta_meters:  float
    mean_rotation_delta_radians: float
    object_count_movable:        int
    improvement_score:           float | None    = None
    per_object_details:          dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializes metrics to a JSON-compatible dictionary."""
        return {
            "scene_name":                  self.scene_name,
            "pipeline_step":               self.pipeline_step,
            "mean_position_delta_meters":  round(self.mean_position_delta_meters,  4),
            "mean_rotation_delta_radians": round(self.mean_rotation_delta_radians, 4),
            "object_count_movable":        self.object_count_movable,
            "improvement_score": (
                round(self.improvement_score, 4)
                if self.improvement_score is not None
                else None
            ),
            "per_object_details": self.per_object_details,
        }

    def summary_line(self) -> str:
        """Returns a compact human-readable summary of the main metrics."""
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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _euclidean_distance_2d(loc_a: list[float], loc_b: list[float]) -> float:
    """
    Euclidean distance in the XY plane between two 3D positions.

    The Z coordinate is ignored: only horizontal displacement is measured.
    """
    dx = loc_a[0] - loc_b[0]
    dy = loc_a[1] - loc_b[1]
    return math.sqrt(dx * dx + dy * dy)


def _angular_difference_z(rot_a: list[float], rot_b: list[float]) -> float:
    """
    Minimum angular difference on the Z axis between two XYZ Euler rotations.

    The result is normalized to [0, pi] to handle equivalent angles
    (e.g. 0 and 2*pi, or pi and -pi).
    """
    diff = abs(rot_a[2] - rot_b[2]) % (2.0 * math.pi)
    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return diff


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_metrics(
    reference_state:   SceneState,
    evaluated_state:   SceneState,
    disordered_state:  SceneState | None = None,
) -> SceneMetrics:
    """
    Computes quality metrics for a given scene state.

    Compares the evaluated state against the reference (original) state to
    measure how closely the reorganization matches the ground truth.

    Args:
        reference_state:  Original scene state (ground truth).
        evaluated_state:  State being evaluated (reorganized or refined).
        disordered_state: Optional disordered state used to compute the
                          improvement_score.

    Returns:
        SceneMetrics with all computed values.
    """
    position_deltas:  list[float] = []
    rotation_deltas:  list[float] = []
    disordered_deltas: list[float] = []
    per_object:       dict[str, dict] = {}

    for ref_obj in reference_state.movable_objects:
        eval_obj = evaluated_state.get(ref_obj.name)
        if eval_obj is None or not eval_obj.is_movable:
            continue

        pos_delta = _euclidean_distance_2d(
            ref_obj.transform.location, eval_obj.transform.location
        )
        rot_delta = _angular_difference_z(
            ref_obj.transform.rotation_euler, eval_obj.transform.rotation_euler
        )

        position_deltas.append(pos_delta)
        rotation_deltas.append(rot_delta)

        obj_details: dict = {
            "position_delta_meters":  round(pos_delta, 4),
            "rotation_delta_degrees": round(math.degrees(rot_delta), 2),
        }

        if disordered_state is not None:
            dis_obj = disordered_state.get(ref_obj.name)
            if dis_obj is not None:
                dis_delta = _euclidean_distance_2d(
                    ref_obj.transform.location, dis_obj.transform.location
                )
                disordered_deltas.append(dis_delta)
                obj_details["disordered_delta_meters"] = round(dis_delta, 4)

        per_object[ref_obj.name] = obj_details

    if not position_deltas:
        logger.warning(
            "No movable objects in common between reference and evaluated states. "
            "Metrics cannot be computed."
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

    improvement_score: float | None = None
    if disordered_deltas:
        if len(disordered_deltas) == len(position_deltas):
            mean_dis_delta = sum(disordered_deltas) / len(disordered_deltas)
            if mean_dis_delta > 0.0:
                raw_score     = 1.0 - (mean_pos_delta / mean_dis_delta)
                improvement_score = max(0.0, min(1.0, raw_score))
        else:
            logger.warning(
                "Cannot compute improvement_score: disordered_state object count (%d) "
                "differs from reference_state (%d).",
                len(disordered_deltas),
                len(position_deltas),
            )

    metrics = SceneMetrics(
        scene_name=evaluated_state.scene_name,
        pipeline_step=evaluated_state.pipeline_step,
        mean_position_delta_meters=mean_pos_delta,
        mean_rotation_delta_radians=mean_rot_delta,
        object_count_movable=len(position_deltas),
        improvement_score=improvement_score,
        per_object_details=per_object,
    )
    logger.info("Metrics for '%s': %s.", evaluated_state.scene_name, metrics.summary_line())
    return metrics


def compute_pipeline_metrics(
    original_state:   SceneState,
    randomized_state: SceneState,
    reordered_state:  SceneState,
    refined_state:    SceneState,
) -> dict[str, SceneMetrics]:
    """
    Computes metrics for all pipeline states in a single call.

    Args:
        original_state:   Original scene (ground truth).
        randomized_state: State after randomization.
        reordered_state:  State after LLM reorganization.
        refined_state:    State after visual-critic refinement.

    Returns:
        Dictionary mapping step name to SceneMetrics.
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

    logger.info("Pipeline metrics for scene '%s':", original_state.scene_name)
    for step, m in results.items():
        logger.info("  %s", m.summary_line())

    return results