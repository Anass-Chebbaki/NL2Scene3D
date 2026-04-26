# scripts/batch_pipeline.py
"""
Esecuzione batch della pipeline NL2Scene3D su piu' scene.

Itera su tutti i file .blend presenti nella directory delle scene originali
e lancia l'elaborazione per ciascuno di essi, raccogliendo le metriche
aggregate al termine.

Utilizzo:
    blender --background --python scripts/batch_pipeline.py -- \\
        --scenes-dir scenes/originals \\
        --outputs-dir scenes/outputs

Nota:
    Richiede che le scene siano file .blend validi e che la variabile
    GEMINI_API_KEY sia configurata nel file .env.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from types import ModuleType

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nl2scene3d.config import get_config
from nl2scene3d.logging_setup import setup_logging

logger = logging.getLogger("nl2scene3d.batch")


def _import_run_pipeline() -> ModuleType:
    """
    Importa il modulo run_pipeline tramite importlib per evitare dipendenze
    da strutture di package non garantite nella directory scripts/.

    Returns:
        Modulo run_pipeline importato.

    Raises:
        ImportError: Se il file run_pipeline.py non esiste.
    """
    module_path = Path(__file__).resolve().parent / "run_pipeline.py"
    if not module_path.exists():
        raise ImportError(
            f"Modulo run_pipeline.py non trovato in: {module_path}"
        )
    spec = importlib.util.spec_from_file_location("run_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossibile creare spec per: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def parse_args() -> argparse.Namespace:
    """
    Parsa gli argomenti da riga di comando per l'esecuzione batch.

    Returns:
        Namespace con gli argomenti parsati.
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="NL2Scene3D Batch Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Directory contenente i file .blend (override config).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Directory radice output (override config).",
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
        help="Salta la chiamata LLM Vision.",
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
    )
    parser.add_argument(
        "--scene-pattern",
        type=str,
        default="*.blend",
        help="Pattern glob per selezionare i file .blend.",
    )
    return parser.parse_args(argv)


def run_single_scene(
    blend_path: Path,
    output_dir: Path,
    seed: int,
    skip_vision: bool,
    max_objects: int,
    log_level: str,
    run_pipeline_fn: object,
) -> dict:
    """
    Lancia la pipeline per una singola scena .blend.

    Args:
        blend_path: Percorso al file .blend da elaborare.
        output_dir: Directory di output per questa scena.
        seed: Seed per la randomizzazione.
        skip_vision: Se True, salta la chiamata LLM Vision.
        max_objects: Limite al numero di oggetti movibili.
        log_level: Livello di logging.

    Returns:
        Dizionario con 'scene', 'status', 'duration_seconds', 'error'.
    """
    run_pipeline = run_pipeline_fn

    scene_name = blend_path.stem
    start_time = time.monotonic()

    logger.info("Avvio elaborazione per scena: %s", scene_name)

    try:
        try:
            import bpy  # noqa: PLC0415
            bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        except Exception as exc:
            logger.error(
                "Impossibile aprire il file .blend '%s': %s", blend_path, exc
            )
            raise RuntimeError(
                f"Errore nell'apertura del file Blender: {exc}"
            ) from exc

        mock_args = argparse.Namespace(
            scene_name=scene_name,
            output_dir=output_dir,
            prompts_dir=None,
            seed=seed,
            skip_vision=skip_vision,
            max_objects=max_objects,
            log_level=log_level,
        )
        run_pipeline(mock_args)

        duration = time.monotonic() - start_time
        return {
            "scene": scene_name,
            "status": "success",
            "duration_seconds": round(duration, 2),
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start_time
        logger.error(
            "Errore durante l'elaborazione di '%s': %s", scene_name, exc
        )
        return {
            "scene": scene_name,
            "status": "error",
            "duration_seconds": round(duration, 2),
            "error": str(exc),
        }


def main() -> None:
    """Punto di ingresso per l'esecuzione batch."""
    try:
        app_config = get_config()
    except EnvironmentError as exc:
        print(
            f"CRITICAL: Configurazione non valida: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args()

    log_level = args.log_level or app_config.logging.level
    setup_logging(level=log_level, logger_name="nl2scene3d.batch")

    scenes_dir: Path = args.scenes_dir or app_config.pipeline.scenes_dir
    outputs_dir: Path = args.outputs_dir or app_config.pipeline.outputs_dir
    seed: int = (
        args.seed if args.seed is not None else app_config.pipeline.randomizer_seed
    )
    max_objects: int = (
        args.max_objects
        if args.max_objects is not None
        else app_config.pipeline.max_movable_objects
    )

    logger.info("=" * 60)
    logger.info("NL2Scene3D - Esecuzione Batch")
    logger.info("Scenes dir: %s", scenes_dir)
    logger.info("Outputs dir: %s", outputs_dir)
    logger.info("=" * 60)

    if not scenes_dir.exists():
        logger.critical(
            "Directory delle scene non trovata: %s", scenes_dir
        )
        sys.exit(1)

    blend_files = sorted(scenes_dir.glob(args.scene_pattern))
    if not blend_files:
        logger.warning(
            "Nessun file .blend trovato in '%s' con pattern '%s'.",
            scenes_dir,
            args.scene_pattern,
        )
        sys.exit(0)

    results: list[dict] = []
    total_start = time.monotonic()

    try:
        run_pipeline_module = _import_run_pipeline()
        run_pipeline_fn = run_pipeline_module.run_pipeline
    except ImportError as exc:
        logger.critical("Impossibile importare run_pipeline: %s", exc)
        sys.exit(1)

    for index, blend_path in enumerate(blend_files, start=1):
        logger.info(
            "--- Scena %d/%d: %s ---", index, len(blend_files), blend_path.name
        )

        scene_output_dir = outputs_dir / blend_path.stem
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_single_scene(
            blend_path=blend_path,
            output_dir=scene_output_dir,
            seed=seed,
            skip_vision=args.skip_vision,
            max_objects=max_objects,
            log_level=log_level,
            run_pipeline_fn=run_pipeline_fn,
        )
        results.append(result)

        # Pausa di cortesia tra scene consecutive per rispettare i rate limit API.
        if index < len(blend_files):
            time.sleep(2)

    total_duration = time.monotonic() - total_start

    report = {
        "total_scenes": len(blend_files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_duration_seconds": round(total_duration, 2),
        "scenes": results,
    }

    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "batch_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Batch completato. Successi: %d/%d. Report: %s",
        report["successful"],
        len(blend_files),
        report_path,
    )


if __name__ == "__main__":
    main()