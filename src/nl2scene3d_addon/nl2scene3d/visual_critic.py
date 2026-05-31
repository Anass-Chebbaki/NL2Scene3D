# nl2scene3d/visual_critic.py
"""
Visual critique of the reorganized layout via LLM Vision.

This module analyzes multi-view renders of the reorganized scene and
produces a list of suggested corrections to improve the layout.

Improvements over the previous version:
    - Multi-view support (4 images: top, iso, iso2, front).
    - Good-layout protection: score >= 8 suppresses all corrections.
    - Post-correction validation with collision checking.
    - Maximum displacement limit per correction to prevent catastrophic moves.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

from nl2scene3d.config import PipelineConfig
from nl2scene3d.gemini_client import GeminiClient, GeminiParsingError
from nl2scene3d.models import LLMCorrection, RoomBounds, SceneObject, SceneState, Transform
from nl2scene3d.utils.geometry import snap_rotation_90

logger = logging.getLogger(__name__)

# Minimum quality score above which corrections are still applied
# to further optimize the layout.
_DEFAULT_MIN_QUALITY_SCORE: int = 7

# Quality score above which the layout is considered good and corrections
# are suppressed to avoid making it worse.
_DEFAULT_GOOD_QUALITY_SCORE: int = 8

# Default maximum number of corrections to apply per iteration.
_DEFAULT_MAX_CORRECTIONS: int = 5

# Maximum allowed displacement for a single correction (meters).
# Prevents catastrophic moves that would destroy the layout.
_MAX_CORRECTION_DISPLACEMENT: float = 1.5


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------

def _parse_corrections_from_llm(
    llm_output: dict | list,
) -> tuple[int, str, list[LLMCorrection]]:
    """
    Parse the raw LLM Vision output into a structured list of corrections.

    Args:
        llm_output: Raw JSON output from the vision model.

    Returns:
        Tuple of (score, quality_assessment, corrections).
    """
    if isinstance(llm_output, list):
        logger.warning(
            "Vision output is a list instead of a dictionary. "
            "Interpreting as a direct list of corrections."
        )
        corrections = [
            LLMCorrection.from_dict(item)
            for item in llm_output
            if isinstance(item, dict)
        ]
        return 5, "Parsed from list format", corrections

    if not isinstance(llm_output, dict):
        logger.error(
            "Vision output is not a valid dictionary (type=%s). No corrections applied.",
            type(llm_output),
        )
        return 5, "Invalid format", []

    score:              int  = int(llm_output.get("score", 5))
    quality_assessment: str  = llm_output.get("quality_assessment", "")
    raw_corrections:    list = llm_output.get("corrections", [])

    corrections: list[LLMCorrection] = []
    for item in raw_corrections:
        if not isinstance(item, dict):
            continue
        try:
            correction = LLMCorrection.from_dict(item)
            corrections.append(correction)
        except (KeyError, TypeError) as exc:
            logger.warning("Invalid correction ignored: %s. Error: %s", item, exc)

    logger.info(
        "Visual critique: score=%d/10, assessment='%s', suggested corrections=%d.",
        score,
        quality_assessment,
        len(corrections),
    )
    return score, quality_assessment, corrections


# ---------------------------------------------------------------------------
# Correction validation
# ---------------------------------------------------------------------------

def _validate_correction_displacement(
    correction: LLMCorrection,
    original_obj: SceneObject,
    max_displacement: float = _MAX_CORRECTION_DISPLACEMENT,
) -> bool:
    """
    Verify that a correction does not move the object beyond the allowed limit.

    Prevents catastrophic displacements that would destroy the layout.

    Args:
        correction:      Correction to validate.
        original_obj:    Object at its current position.
        max_displacement: Maximum allowed displacement in meters.

    Returns:
        True if the correction is within limits, False otherwise.
    """
    if correction.new_location and len(correction.new_location) == 3:
        dx = correction.new_location[0] - original_obj.transform.location[0]
        dy = correction.new_location[1] - original_obj.transform.location[1]
        displacement = math.sqrt(dx * dx + dy * dy)
        if displacement > max_displacement:
            logger.warning(
                "Correction for '%s' rejected: displacement of %.2f m exceeds limit of %.2f m.",
                correction.object_name,
                displacement,
                max_displacement,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------

def _apply_corrections_to_state(
    state: SceneState,
    corrections: list[LLMCorrection],
    max_corrections: int,
    room_bounds: Optional[RoomBounds] = None,
) -> SceneState:
    """
    Apply the corrections suggested by the LLM Vision to the scene state.

    Args:
        state:           Current scene state to update.
        corrections:     List of corrections to apply.
        max_corrections: Maximum number of corrections to apply.
        room_bounds:     Bounds used to clamp coordinates.

    Returns:
        New SceneState with the corrections applied.
    """
    # In Python 3.7+ dicts preserve insertion order, so the original object
    # order from state.objects is maintained. Objects not targeted by any
    # correction remain in the dict and are extracted normally at the end.
    objects_by_name: dict[str, SceneObject] = {
        obj.name: obj.copy() for obj in state.objects
    }

    applied_count = 0
    skipped_count = 0

    corrections_to_apply = corrections[:max_corrections]
    if len(corrections) > max_corrections:
        logger.info(
            "Corrections capped from %d to %d to avoid over-modifying the layout.",
            len(corrections),
            max_corrections,
        )

    for correction in corrections_to_apply:
        target_obj = objects_by_name.get(correction.object_name)

        if target_obj is None:
            logger.warning(
                "Correction ignored: object '%s' not found in the scene.",
                correction.object_name,
            )
            skipped_count += 1
            continue

        if not target_obj.is_movable:
            logger.warning(
                "Correction ignored: object '%s' is not movable.",
                correction.object_name,
            )
            skipped_count += 1
            continue

        if not _validate_correction_displacement(correction, target_obj):
            skipped_count += 1
            continue

        new_location = list(target_obj.transform.location)
        new_rotation = list(target_obj.transform.rotation_euler)

        if correction.action in ("move", "move_and_rotate"):
            if correction.new_location and len(correction.new_location) == 3:
                candidate_location = list(correction.new_location)
                # Always preserve the original Z value.
                candidate_location[2] = target_obj.transform.location[2]
                if room_bounds is not None:
                    candidate_location = room_bounds.clamp_location(candidate_location)
                new_location = candidate_location

        if correction.action in ("rotate", "move_and_rotate"):
            if (
                correction.new_rotation_euler
                and len(correction.new_rotation_euler) == 3
            ):
                raw_rotation = list(correction.new_rotation_euler)
                new_rotation = [
                    target_obj.transform.rotation_euler[0],
                    target_obj.transform.rotation_euler[1],
                    snap_rotation_90(raw_rotation[2]),
                ]

        target_obj.transform = Transform(
            location=new_location,
            rotation_euler=new_rotation,
            dimensions=list(target_obj.transform.dimensions),
            origin_offset=list(target_obj.transform.origin_offset),
        )

        logger.debug(
            "Correction applied to '%s': action=%s, location=%s, rotation=%s.",
            correction.object_name,
            correction.action,
            new_location,
            new_rotation,
        )
        applied_count += 1

    # Post-validation: resolve collisions introduced by the corrections.
    from nl2scene3d.utils.geometry import has_collision

    final_objects: list[SceneObject] = []

    # Add all non-movable (structural) objects first.
    for obj in objects_by_name.values():
        if not obj.is_movable:
            final_objects.append(obj)

    # Add movable objects one by one, checking for collisions.
    collision_reverted = 0
    for obj in objects_by_name.values():
        if not obj.is_movable:
            continue

        main_furniture_categories = {
            "bed", "table", "storage", "seating_large", "seating_small", "furniture"
        }
        if obj.category in main_furniture_categories:
            collidable_objects = [
                o for o in final_objects
                if o.category in main_furniture_categories or o.category == "structural"
            ]
            if has_collision(obj, collidable_objects, check_walls=True):
                # The correction introduced a collision: revert to the original position.
                original_obj = state.get(obj.name)
                if original_obj is not None:
                    obj.transform = original_obj.transform.copy()
                    collision_reverted += 1
                    logger.warning(
                        "Correction for '%s' reverted: new position causes a collision.",
                        obj.name,
                    )

        final_objects.append(obj)

    if collision_reverted > 0:
        logger.info(
            "Post-validation: %d corrections reverted due to collisions.",
            collision_reverted,
        )

    logger.info(
        "Corrections applied: %d of %d requested (%d skipped, %d reverted for collisions).",
        applied_count - collision_reverted,
        len(corrections_to_apply),
        skipped_count,
        collision_reverted,
    )

    return SceneState(
        scene_name=state.scene_name,
        objects=final_objects,
        room_bounds=state.room_bounds,
        pipeline_step="refined",
        metadata={
            "applied_corrections": applied_count - collision_reverted,
            "skipped_corrections": skipped_count,
            "collision_reverted":  collision_reverted,
        },
    )


# ---------------------------------------------------------------------------
# VisualCritic
# ---------------------------------------------------------------------------

class VisualCritic:
    """
    Performs visual critique of the reorganized layout via LLM Vision.

    Attributes:
        client:              Gemini client for API calls.
        prompts_dir:         Directory containing prompt templates.
        config:              Pipeline configuration (optional).
        min_quality_score:   Minimum score above which corrections are applied.
        good_quality_score:  Score above which the layout is considered good
                             and corrections are suppressed.
        max_corrections:     Maximum number of corrections per iteration.
    """

    def __init__(
        self,
        client: GeminiClient,
        prompts_dir: Path,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """
        Initialize the visual critic.

        Args:
            client:      Configured Gemini client.
            prompts_dir: Directory containing prompt templates.
            config:      Pipeline configuration for thresholds and limits.
                         Default constants are used when None.
        """
        self.client      = client
        self.prompts_dir = prompts_dir
        self.config      = config

        if config is not None:
            self.min_quality_score  = config.min_quality_score
            self.good_quality_score = config.good_quality_score
            self.max_corrections    = config.max_corrections
        else:
            self.min_quality_score  = _DEFAULT_MIN_QUALITY_SCORE
            self.good_quality_score = _DEFAULT_GOOD_QUALITY_SCORE
            self.max_corrections    = _DEFAULT_MAX_CORRECTIONS

        logger.info(
            "VisualCritic initialized. "
            "min_quality_score=%d, good_quality_score=%d, max_corrections=%d.",
            self.min_quality_score,
            self.good_quality_score,
            self.max_corrections,
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _build_critic_prompt(self, state: SceneState, num_views: int = 1) -> str:
        """
        Build the visual critique prompt by loading and filling the template.

        Args:
            state:     Reorganized scene state.
            num_views: Number of image views attached to the request.

        Returns:
            Formatted prompt string with room information injected.

        Raises:
            FileNotFoundError: If the prompt template does not exist.
        """
        template_path = self.prompts_dir / "critic_user.txt"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Visual critique prompt template not found: {template_path}"
            )

        with open(template_path, encoding="utf-8") as fh:
            template = fh.read()

        movable_names = [obj.name for obj in state.movable_objects]
        names_str     = ", ".join(movable_names)

        # Append a strict instruction to prevent the LLM from inventing object names.
        instruction = (
            f"\n\nCRITICAL INSTRUCTION: When suggesting corrections, you MUST ONLY use "
            f"exact object names from this list: [{names_str}]. "
            f"Do NOT invent generic names like 'computer_case' or 'office_chair'. "
            f"You must copy the exact internal name from this list to ensure the code "
            f"can find the object."
        )

        room_bounds = state.room_bounds
        if room_bounds is not None:
            prompt = template.format(
                x_min=room_bounds.x_min,
                x_max=room_bounds.x_max,
                y_min=room_bounds.y_min,
                y_max=room_bounds.y_max,
            )
        else:
            prompt = template.format(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0)

        return prompt + instruction

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def critique_and_refine(
        self,
        reordered_state: SceneState,
        render_iso_path: Path,
        render_paths: Optional[dict[str, Path]] = None,
    ) -> SceneState:
        """
        Analyze scene renders and apply the suggested corrections.

        When render_paths contains multiple views, a multi-image API call is
        used to give the model a complete view of the scene. Otherwise a single
        isometric view is used for backward compatibility.

        If the model assigns a score >= good_quality_score (default 8),
        corrections are suppressed: the layout is already good.

        Args:
            reordered_state: Scene state after LLM reorganization.
            render_iso_path: Path to the isometric render (single-view fallback).
            render_paths:    Optional dictionary with all available view paths.

        Returns:
            Refined SceneState, or the original state unchanged if corrections
            cannot be applied or the LLM call fails.
        """
        logger.info(
            "Starting visual critique for scene '%s'. Render: %s",
            reordered_state.scene_name,
            render_iso_path,
        )

        # Collect image paths to send, in the preferred order.
        image_paths: list[Path] = []
        if render_paths:
            for key in ["top", "iso", "iso2", "front"]:
                if key in render_paths and render_paths[key].exists():
                    image_paths.append(render_paths[key])

        if not image_paths:
            # Fallback to the single isometric view.
            image_paths = [render_iso_path]

        num_views   = len(image_paths)
        user_prompt = self._build_critic_prompt(reordered_state, num_views)

        try:
            if num_views > 1:
                logger.info(
                    "Sending %d views to the vision model: %s",
                    num_views,
                    [p.name for p in image_paths],
                )
                llm_output = self.client.call_vision_multi(image_paths, user_prompt)
            else:
                llm_output = self.client.call_vision(image_paths[0], user_prompt)

        except GeminiParsingError as exc:
            logger.error(
                "Vision response parsing failed: %s. Returning the reordered state unchanged.",
                exc,
            )
            refined = reordered_state.copy()
            refined.pipeline_step = "refined"
            refined.metadata = {"error": str(exc), "applied_corrections": 0}
            return refined

        score, quality_assessment, corrections = _parse_corrections_from_llm(llm_output)

        logger.info(
            "Visual quality score: %d/10. Assessment: '%s'.",
            score,
            quality_assessment,
        )

        if not corrections:
            logger.info("No corrections suggested. Layout considered optimal.")
            refined = reordered_state.copy()
            refined.pipeline_step = "refined"
            refined.metadata = {
                "quality_score":      score,
                "quality_assessment": quality_assessment,
                "applied_corrections": 0,
            }
            return refined

        # Protect good layouts: suppress corrections when the score is high.
        if score >= self.good_quality_score:
            logger.info(
                "Score %d >= good quality threshold %d. "
                "Corrections suppressed to protect the existing layout. "
                "(%d suggestions ignored)",
                score,
                self.good_quality_score,
                len(corrections),
            )
            refined = reordered_state.copy()
            refined.pipeline_step = "refined"
            refined.metadata = {
                "quality_score":         score,
                "quality_assessment":    quality_assessment,
                "applied_corrections":   0,
                "corrections_suppressed": len(corrections),
                "reason":                "Score above good_quality_score threshold",
            }
            return refined

        if score >= self.min_quality_score:
            logger.info(
                "Score %d >= min threshold %d but < good threshold %d. "
                "Applying corrections cautiously.",
                score,
                self.min_quality_score,
                self.good_quality_score,
            )
        else:
            logger.info(
                "Score %d < threshold %d. Applying corrections to recover insufficient quality.",
                score,
                self.min_quality_score,
            )

        refined_state = _apply_corrections_to_state(
            reordered_state,
            corrections,
            self.max_corrections,
            reordered_state.room_bounds,
        )
        refined_state.metadata["quality_score"]      = score
        refined_state.metadata["quality_assessment"] = quality_assessment

        return refined_state