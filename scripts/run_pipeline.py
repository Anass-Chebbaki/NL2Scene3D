# scripts/run_pipeline.py
"""
Orchestratore end-to-end della pipeline NL2Scene3D.

Coordina l'esecuzione del flusso di lavoro:
- Caricamento e introspezione della scena .blend
- Render originale (top-down + isometrico)
- Randomizzazione degli oggetti
- Render + estrazione stato disordinato
- Chiamata LLM: riordino testuale
- Applicazione delle coordinate e render
- Chiamata LLM Vision: critica visiva
- Applicazione delle rifiniture e render finale

Utilizzo:
    blender --background scene.blend --python scripts/run_pipeline.py -- \\
        --scene-name my_scene \\
        --output-dir scenes/outputs/my_scene

Nota:
    Questo script deve essere eseguito tramite il Python integrato in
    Blender, non tramite il Python di sistema.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
import platform
import sys

if platform.system() == "Windows":
    _VENV_SITE_PACKAGES = _PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
else:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _VENV_SITE_PACKAGES = _PROJECT_ROOT / ".venv" / "lib" / version / "site-packages"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Aggiunge le dipendenze del .venv al Python di Blender (che e' isolato).
if _VENV_SITE_PACKAGES.exists() and str(_VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE_PACKAGES))

from nl2scene3d.blender.renderer import BlenderRenderer
from nl2scene3d.config import get_config, RandomizerConfig
from nl2scene3d.gemini_client import GeminiClient
from nl2scene3d.randomizer import SceneRandomizer
from nl2scene3d.scene_applicator import SceneApplicator
from nl2scene3d.scene_loader import SceneLoader
from nl2scene3d.scene_reorganizer import SceneReorganizer
from nl2scene3d.visual_critic import VisualCritic
from nl2scene3d.metrics import compute_pipeline_metrics
from nl2scene3d.logging_setup import setup_logging

logger = logging.getLogger("nl2scene3d.pipeline")


def parse_args() -> argparse.Namespace:
    """
    Parsa gli argomenti da riga di comando.

    Gli argomenti dopo '--' in Blender vengono passati allo script Python.

    Returns:
        Namespace con gli argomenti parsati.
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="NL2Scene3D Pipeline - Scene reorganization via MLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        required=True,
        help="Nome identificativo della scena (usato nei nomi dei file output).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory dove salvare JSON e render per questa scena.",
    )
    parser.add_argument(
        "--blend-file",
        type=Path,
        default=None,
        help="Percorso al file .blend da elaborare.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Directory dei template dei prompt per Gemini.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed per la randomizzazione (override config).",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Salta la seconda chiamata LLM Vision.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Numero massimo di oggetti movibili (override config).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Livello di verbosita' del logging (override config).",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Esegue la pipeline completa per una scena.

    Args:
        args: Argomenti da riga di comando parsati.
    """
    try:
        app_config = get_config()
    except EnvironmentError as exc:
        print(f"CRITICAL: Errore di configurazione: {exc}", file=sys.stderr)
        sys.exit(1)

    log_level = args.log_level or app_config.logging.level
    setup_logging(level=log_level)

    logger.info("=" * 60)
    logger.info("NL2Scene3D Pipeline - Avvio")
    logger.info("Scena: %s", args.scene_name)
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 60)

    if args.blend_file:
        import bpy
        try:
            bpy.ops.wm.open_mainfile(filepath=str(args.blend_file))
        except Exception as exc:
            logger.error("Impossibile aprire il file .blend '%s': %s", args.blend_file, exc)
            sys.exit(1)

    # Use a copy to avoid mutating the global configuration singleton
    import copy
    pipeline_config = copy.deepcopy(app_config.pipeline)
    if args.max_objects is not None:
        pipeline_config.max_movable_objects = args.max_objects

    prompts_dir: Path = args.prompts_dir or (_PROJECT_ROOT / "config" / "prompts")
    seed: int = (
        args.seed if args.seed is not None else pipeline_config.randomizer_seed
    )

    gemini_client = GeminiClient(app_config.gemini)
    loader = SceneLoader(config=pipeline_config)

    rand_config = RandomizerConfig(
        seed=seed,
        jitter_ratio=pipeline_config.jitter_ratio,
        rotate_z_only=True,
        check_overlaps=True,
        wall_margin=pipeline_config.wall_margin,
        max_overlap_ratio=pipeline_config.max_overlap_ratio,
        max_placement_attempts=pipeline_config.max_placement_attempts,
    )
    randomizer = SceneRandomizer(config=rand_config)

    reorganizer = SceneReorganizer(
        client=gemini_client,
        prompts_dir=prompts_dir,
    )

    applicator = SceneApplicator()

    renderer = BlenderRenderer(
        output_dir=args.output_dir,
        config=app_config.render,
    )

    visual_critic: VisualCritic | None = None
    if not args.skip_vision:
        visual_critic = VisualCritic(
            client=gemini_client,
            prompts_dir=prompts_dir,
            config=pipeline_config,
        )

    # -------------------------------------------------------------------------
    # Esecuzione della pipeline
    # -------------------------------------------------------------------------

    logger.info("Estrazione stato originale.")
    original_state = loader.extract_scene_state(scene_name=args.scene_name)
    loader.save_state_to_json(original_state, args.output_dir / "scene_original.json")

    logger.info("Render originale.")
    renderer.render_step(step_name="original", state=original_state, quality="preview")

    logger.info("Randomizzazione.")
    randomized_state = randomizer.randomize(original_state)
    applicator.apply_state(randomized_state)
    loader.save_state_to_json(
        randomized_state, args.output_dir / "scene_randomized.json"
    )

    logger.info("Render scena disordinata.")
    randomized_renders = renderer.render_step(
        step_name="randomized", state=randomized_state, quality="preview"
    )

    logger.info("Chiamata LLM per riordino testuale.")
    reordered_state = reorganizer.reorganize(randomized_state)
    # Forza l'uso dei bounds originali per evitare spostamenti della camera
    reordered_state.room_bounds = original_state.room_bounds
    
    if reordered_state.pipeline_step == "reordered_failed":
        logger.warning("Il riordino ha fallito. Utilizzo stato randomizzato per i prossimi step. Salto critica visiva.")
        args.skip_vision = True
    
    loader.save_state_to_json(
        reordered_state, args.output_dir / "scene_reordered.json"
    )

    logger.info("Applicazione coordinate riordinate e render.")
    applicator.apply_state(reordered_state)
    reordered_renders = renderer.render_step(
        step_name="reordered", state=reordered_state, quality="preview"
    )

    logger.info("Critica visiva.")
    if args.skip_vision or visual_critic is None:
        refined_state = reordered_state.copy()
        refined_state.pipeline_step = "refined"
    else:
        refined_state = visual_critic.critique_and_refine(
            reordered_state=reordered_state,
            render_iso_path=reordered_renders["iso"],
        )
        # Forza l'uso dei bounds originali anche dopo la rifinitura visiva
        refined_state.room_bounds = original_state.room_bounds

    loader.save_state_to_json(refined_state, args.output_dir / "scene_refined.json")

    if refined_state.metadata.get("applied_corrections", 0) > 0:
        logger.info(
            "Applicazione di %d correzioni visive.",
            refined_state.metadata["applied_corrections"],
        )
        applicator.apply_state(refined_state)

    logger.info("Render finale ad alta qualita'.")
    renderer.render_step(step_name="final", state=refined_state, quality="final")

    logger.info("Calcolo delle metriche.")
    metrics = compute_pipeline_metrics(
        original_state=original_state,
        randomized_state=randomized_state,
        reordered_state=reordered_state,
        refined_state=refined_state,
    )
    
    metrics_dict = {step: m.to_dict() for step, m in metrics.items()}
    import json
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    logger.info("Salvataggio file .blend finale.")
    applicator.save_blend_file(args.output_dir / f"{args.scene_name}_final.blend")

    logger.info("=" * 60)
    logger.info("Pipeline completata con successo. Output: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    _args = parse_args()
    run_pipeline(_args)