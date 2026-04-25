"""
Esecuzione batch della pipeline NL2Scene3D su piu' scene.

Questo script itera su tutti i file .blend presenti nella directory
delle scene originali e lancia la pipeline per ciascuno di essi,
raccogliendo le metriche aggregate al termine.

UTILIZZO:
    blender --background --python scripts/batch_pipeline.py -- \
        --scenes-dir scenes/originals \
        --outputs-dir scenes/outputs \
        --seed 42

NOTA: Richiede che le scene siano file .blend validi e che
      la variabile GEMINI_API_KEY sia configurata nel file .env.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nl2scene3d.config import get_config
from nl2scene3d.logging_setup import setup_logging

logger = logging.getLogger("nl2scene3d.batch")

def parse_args() -> argparse.Namespace:
    """Parsa gli argomenti da riga di comando per l'esecuzione batch."""
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
        default=_PROJECT_ROOT / "scenes" / "originals",
        help="Directory contenente i file .blend da processare.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=_PROJECT_ROOT / "scenes" / "outputs",
        help="Directory radice dove salvare gli output per ogni scena.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=_PROJECT_ROOT / "config" / "prompts",
        help="Directory dei template dei prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed fisso per la randomizzazione (garantisce riproducibilita').",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Salta la chiamata LLM Vision per tutti le scene.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=20,
        help="Numero massimo di oggetti movibili per scena.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Percorso opzionale al file di log.",
    )
    parser.add_argument(
        "--scene-pattern",
        type=str,
        default="*.blend",
        help="Pattern glob per selezionare i file .blend da processare.",
    )
    return parser.parse_args(argv)

def run_single_scene(
    blend_path: Path,
    output_dir: Path,
    prompts_dir: Path,
    seed: int,
    skip_vision: bool,
    max_objects: int,
) -> dict:
    """
    Lancia la pipeline per una singola scena .blend.

    Importa e chiama run_pipeline da run_pipeline.py tramite argparse
    simulato, oppure direttamente come funzione se integrato.

    Args:
        blend_path: Percorso al file .blend.
        output_dir: Directory di output per questa scena.
        prompts_dir: Directory dei template dei prompt.
        seed: Seed per la randomizzazione.
        skip_vision: Se True, salta la chiamata vision.
        max_objects: Numero massimo di oggetti movibili.

    Returns:
        Dizionario con risultato dell'elaborazione:
        {'scene': str, 'status': 'success'|'error', 'duration_seconds': float,
         'error': str|None}
    """
    import argparse as _argparse
    from scripts.run_pipeline import run_pipeline  # noqa: PLC0415

    scene_name = blend_path.stem
    start_time = time.monotonic()

    logger.info(
        "Avvio pipeline per scena: %s (%s)",
        scene_name,
        blend_path,
    )

    try:
        # Costruisce gli argomenti simulati per run_pipeline
        mock_args = _argparse.Namespace(
            scene_name=scene_name,
            output_dir=output_dir,
            prompts_dir=prompts_dir,
            seed=seed,
            skip_vision=skip_vision,
            max_objects=max_objects,
            log_level="INFO",
        )

        # Apertura del file .blend prima di richiamare la pipeline
        try:
            import bpy  # noqa: PLC0415
            bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        except ImportError:
            logger.warning(
                "bpy non disponibile. Il loader usera' la scena gia' aperta."
            )

        run_pipeline(mock_args)
        duration = time.monotonic() - start_time

        logger.info(
            "Scena '%s' completata in %.1f secondi.", scene_name, duration
        )
        return {
            "scene": scene_name,
            "status": "success",
            "duration_seconds": round(duration, 2),
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start_time
        logger.error(
            "Errore durante l'elaborazione di '%s': %s",
            scene_name,
            exc,
            exc_info=True,
        )
        return {
            "scene": scene_name,
            "status": "error",
            "duration_seconds": round(duration, 2),
            "error": str(exc),
        }

def main() -> None:
    """Punto di ingresso per l'esecuzione batch."""
    args = parse_args()

    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        logger_name="nl2scene3d.batch",
    )

    logger.info("=" * 60)
    logger.info("NL2Scene3D - Esecuzione Batch")
    logger.info("Scene dir: %s", args.scenes_dir)
    logger.info("Output dir: %s", args.outputs_dir)
    logger.info("Pattern: %s", args.scene_pattern)
    logger.info("=" * 60)

    # Verifica configurazione
    try:
        get_config()
    except EnvironmentError as exc:
        logger.critical("Configurazione non valida: %s", exc)
        sys.exit(1)

    # Scopri tutti i file .blend da processare
    if not args.scenes_dir.exists():
        logger.critical(
            "Directory delle scene non trovata: %s", args.scenes_dir
        )
        sys.exit(1)

    blend_files = sorted(args.scenes_dir.glob(args.scene_pattern))
    if not blend_files:
        logger.warning(
            "Nessun file .blend trovato in '%s' con pattern '%s'.",
            args.scenes_dir,
            args.scene_pattern,
        )
        sys.exit(0)

    logger.info(
        "Trovati %d file .blend da processare.", len(blend_files)
    )

    # Elabora ogni scena
    results: list[dict] = []
    total_start = time.monotonic()

    for index, blend_path in enumerate(blend_files, start=1):
        logger.info(
            "--- Scena %d/%d: %s ---",
            index,
            len(blend_files),
            blend_path.name,
        )

        scene_output_dir = args.outputs_dir / blend_path.stem
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_single_scene(
            blend_path=blend_path,
            output_dir=scene_output_dir,
            prompts_dir=args.prompts_dir,
            seed=args.seed,
            skip_vision=args.skip_vision,
            max_objects=args.max_objects,
        )
        results.append(result)

        # Piccola pausa tra le scene per rispettare il rate limit API
        if index < len(blend_files):
            logger.debug("Pausa 2 secondi tra una scena e la successiva.")
            time.sleep(2)

    total_duration = time.monotonic() - total_start

    # Salva il report batch
    report_path = args.outputs_dir / "batch_report.json"
    report = {
        "total_scenes": len(blend_files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_duration_seconds": round(total_duration, 2),
        "scenes": results,
    }

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Batch completato.")
    logger.info(
        "Successi: %d/%d. Fallimenti: %d/%d.",
        report["successful"],
        report["total_scenes"],
        report["failed"],
        report["total_scenes"],
    )
    logger.info("Durata totale: %.1f secondi.", total_duration)
    logger.info("Report salvato: %s", report_path)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
