"""
Orchestratore end-to-end della pipeline NL2Scene3D.

Questo script coordina l'esecuzione del flusso di lavoro:
- Caricamento e introspezione della scena .blend
- Render originale (top-down + isometrico)
- Randomizzazione degli oggetti
- Render + estrazione stato disordinato
- Chiamata LLM: riordino testuale
- Applica le coordinate e render
- Chiamata LLM Vision: critica visiva
- Applica le rifiniture e render finale

UTILIZZO:
    blender --background scene.blend --python scripts/run_pipeline.py -- \
        --scene-name my_scene \
        --output-dir scenes/outputs/my_scene

NOTA: Questo script deve essere eseguito tramite il Blender Python
      embedded, non tramite il Python di sistema.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Aggiunge src/ al sys.path per permettere l'import dei moduli del pacchetto
# quando eseguito dall'interno di Blender con --python
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nl2scene3d.blender.renderer import BlenderRenderer, PREVIEW_CONFIG, FINAL_CONFIG
from nl2scene3d.config import get_config
from nl2scene3d.gemini_client import GeminiClient
from nl2scene3d.models import RenderConfig as NL2RenderConfig
from nl2scene3d.randomizer import RandomizerConfig, SceneRandomizer
from nl2scene3d.scene_applicator import SceneApplicator
from nl2scene3d.scene_loader import SceneLoader
from nl2scene3d.scene_reorganizer import SceneReorganizer
from nl2scene3d.visual_critic import VisualCritic

# Configurazione del logging per l'intera pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("nl2scene3d.pipeline")

def parse_args() -> argparse.Namespace:
    """
    Parsa gli argomenti da riga di comando.

    Gli argomenti dopo '--' in Blender vengono passati allo script Python.

    Returns:
        Namespace con gli argomenti parsati.
    """
    # In Blender, gli argomenti dopo '--' sono accessibili tramite sys.argv
    # dopo il separatore '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
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
        "--prompts-dir",
        type=Path,
        default=_PROJECT_ROOT / "config" / "prompts",
        help="Directory dei template dei prompt per Gemini.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed per la randomizzazione (0 = casuale).",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Salta la seconda chiamata LLM Vision. "
             "Utile per test rapidi o per risparmiare quota API.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=20,
        help="Numero massimo di oggetti movibili da includere nel layout.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Livello di verbosita' del logging.",
    )

    return parser.parse_args(argv)

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Esegue la pipeline completa per una scena.

    Args:
        args: Argomenti da riga di comando parsati.
    """
    # Configura il livello di logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("NL2Scene3D Pipeline - Avvio")
    logger.info("Scena: %s", args.scene_name)
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 60)

    # Crea la directory di output per questa scena
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Carica la configurazione dall'ambiente
    # -------------------------------------------------------------------------
    try:
        app_config = get_config()
    except EnvironmentError as exc:
        logger.critical("Errore di configurazione: %s", exc)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Inizializza i componenti della pipeline
    # -------------------------------------------------------------------------
    gemini_client = GeminiClient(app_config.gemini)

    loader = SceneLoader(max_objects=args.max_objects)

    randomizer_config = RandomizerConfig(
        seed=args.seed,
        jitter_ratio=0.8,
        rotate_z_only=True,
        check_overlaps=True,
    )
    randomizer = SceneRandomizer(config=randomizer_config)

    reorganizer = SceneReorganizer(
        client=gemini_client,
        prompts_dir=args.prompts_dir,
    )

    applicator = SceneApplicator()

    renderer = BlenderRenderer(
        output_dir=output_dir,
        preview_config=PREVIEW_CONFIG,
        final_config=FINAL_CONFIG,
    )

    if not args.skip_vision:
        visual_critic = VisualCritic(
            client=gemini_client,
            prompts_dir=args.prompts_dir,
        )

    # =========================================================================
    # Caricamento e introspezione della scena
    # =========================================================================
    logger.info("--- Estrazione stato originale ---")
    original_state = loader.extract_scene_state(scene_name=args.scene_name)

    original_json_path = output_dir / "scene_original.json"
    loader.save_state_to_json(original_state, original_json_path)
    logger.info(
        "Estratti %d oggetti (%d movibili).",
        len(original_state.objects),
        len(original_state.movable_objects),
    )

    # =========================================================================
    # Render originale
    # =========================================================================
    logger.info("--- Render originale ---")
    original_renders = renderer.render_step(
        step_name="original",
        state=original_state,
        quality="preview",
    )
    logger.info("Render originali: %s", original_renders)

    # =========================================================================
    # Randomizzazione
    # =========================================================================
    logger.info("--- Randomizzazione ---")
    randomized_state = randomizer.randomize(original_state)

    # Applica la randomizzazione alla scena Blender
    applicator.apply_state(randomized_state)

    randomized_json_path = output_dir / "scene_randomized.json"
    loader.save_state_to_json(randomized_state, randomized_json_path)

    # =========================================================================
    # Render + estrazione stato disordinato
    # =========================================================================
    logger.info("--- Render scena disordinata ---")
    randomized_renders = renderer.render_step(
        step_name="randomized",
        state=randomized_state,
        quality="preview",
    )
    logger.info("Render disordinati: %s", randomized_renders)

    # =========================================================================
    # Chiamata LLM - Riordino testuale
    # =========================================================================
    logger.info("--- Chiamata LLM per riordino testuale ---")
    reordered_state = reorganizer.reorganize(randomized_state)

    reordered_json_path = output_dir / "scene_reordered.json"
    loader.save_state_to_json(reordered_state, reordered_json_path)

    # =========================================================================
    # Applica le coordinate riordinate e renderizza
    # =========================================================================
    logger.info("--- Applicazione coordinate riordinate ---")
    applicator.apply_state(reordered_state)

    reordered_renders = renderer.render_step(
        step_name="reordered",
        state=reordered_state,
        quality="preview",
    )
    logger.info("Render riordinati: %s", reordered_renders)

    # =========================================================================
    # Chiamata LLM Vision - Critica visiva
    # =========================================================================
    if args.skip_vision:
        logger.info("--- Critica visiva SALTATA (--skip-vision) ---")
        refined_state = reordered_state
        refined_state.pipeline_step = "refined"
    else:
        logger.info("--- Chiamata LLM Vision per critica visiva ---")
        iso_render_path = reordered_renders["iso"]
        refined_state = visual_critic.critique_and_refine(
            reordered_state=reordered_state,
            render_iso_path=iso_render_path,
        )

    refined_json_path = output_dir / "scene_refined.json"
    loader.save_state_to_json(refined_state, refined_json_path)

    # Applica le rifiniture se ci sono state correzioni
    if refined_state.metadata.get("applied_corrections", 0) > 0:
        logger.info(
            "Applicazione di %d correzioni visive.",
            refined_state.metadata["applied_corrections"],
        )
        applicator.apply_state(refined_state)

    # =========================================================================
    # Render finale ad alta qualita'
    # =========================================================================
    logger.info("--- Render finale ad alta qualita' ---")
    final_renders = renderer.render_step(
        step_name="final",
        state=refined_state,
        quality="final",
    )
    logger.info("Render finali: %s", final_renders)

    # =========================================================================
    # Riepilogo
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Pipeline completata con successo.")
    logger.info("Output salvati in: %s", output_dir)
    logger.info("File JSON prodotti:")
    logger.info("  - %s", original_json_path.name)
    logger.info("  - %s", randomized_json_path.name)
    logger.info("  - %s", reordered_json_path.name)
    logger.info("  - %s", refined_json_path.name)
    logger.info("Render prodotti:")
    for step_renders in [original_renders, randomized_renders, reordered_renders, final_renders]:
        for view, path in step_renders.items():
            logger.info("  - %s", path.name)
    logger.info("=" * 60)

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
