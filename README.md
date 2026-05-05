# NL2Scene3D

**Scene Reorganization from Random to Ordered via Multimodal Language Models**

Version 0.1.0 — License: MIT

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture and Design](#architecture-and-design)
3. [Pipeline Description](#pipeline-description)
4. [Project Structure](#project-structure)
5. [Module Reference](#module-reference)
6. [Configuration](#configuration)
7. [Requirements](#requirements)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Output Artifacts](#output-artifacts)
11. [Metrics and Evaluation](#metrics-and-evaluation)
12. [Development and Testing](#development-and-testing)

---

## Overview

NL2Scene3D is a research pipeline that takes pre-existing, fully furnished 3D room scenes in Blender format (`.blend`), artificially disorganizes the layout of movable objects, and uses a Multimodal Large Language Model (MLLM) — specifically Google Gemini — to reorganize the scene according to professional interior design principles. The result is then refined through a second, vision-based LLM pass that analyzes an isometric render of the reordered scene and applies corrective adjustments.

The system is designed to operate entirely within Blender's embedded Python interpreter, leveraging the `bpy` API for scene introspection, object manipulation, and rendering. All interactions with the language model are handled through the Google Generative AI SDK.

The primary use case is research into the spatial reasoning capabilities of multimodal language models: can a model, given a JSON description of a chaotic room layout and knowledge of the room's dimensions, produce a plausible and professionally coherent arrangement?

---

## Architecture and Design

The project follows a modular, pipeline-oriented architecture. Each concern is isolated into a dedicated module within the `src/nl2scene3d/` package. The pipeline is orchestrated by a single script (`scripts/run_pipeline.py`) that instantiates and sequences the individual components.

Key design decisions:

- **Strict separation between Blender-dependent and Blender-independent code.** Modules under `src/nl2scene3d/blender/` require the `bpy` environment and are only imported at runtime within Blender. All other modules (configuration, data models, the Gemini client, the randomizer, the reorganizer) are pure Python and can be tested independently without a Blender installation.

- **Typed dataclasses as the internal data contract.** The `SceneState`, `SceneObject`, `ObjectTransform`, `RoomBounds`, and `LLMCorrection` dataclasses defined in `models.py` serve as the single source of truth for scene data throughout the pipeline. Every module produces and consumes these types, ensuring structural consistency and enabling reliable JSON serialization and deserialization.

- **Configuration with clear precedence.** Application settings are stored in `config/settings.toml`. Sensitive values (the API key) are kept in a `.env` file and never committed. At runtime, environment variables take precedence over TOML values, which take precedence over hardcoded defaults. The configuration is loaded once as a thread-safe singleton (`AppConfig`) and reused throughout the pipeline.

- **Prompt templates as external files.** The prompts sent to Gemini are not hardcoded into the Python source. They are loaded from text files in `config/prompts/`, making it straightforward to iterate on prompt engineering without modifying application code.

- **Reproducibility through seeded randomization.** The artificial disorganization step uses Python's `random.Random` with a configurable seed. Setting a fixed seed produces identical randomized layouts across runs, enabling reproducible experiments.

---

## Pipeline Description

The pipeline executes the following steps in sequence. Each step produces one or more output artifacts (JSON state files and rendered images) in the designated output directory.

### Step 1 — Scene Loading and State Extraction

The Blender scene (`.blend` file) is opened via `bpy.ops.wm.open_mainfile`. The `SceneLoader` iterates over all objects in the active collection and classifies each one as movable or non-movable based on a set of configurable rules:

- Objects of non-mesh types (cameras, lights, armatures, empties, curves) are always non-movable.
- Objects whose names match structural patterns (wall, floor, ceiling, door, window, and Italian equivalents) are classified as structural and non-movable.
- Ceiling-mounted light fixtures are non-movable.
- Objects whose largest dimension is below the configured minimum (default: 5 cm) are treated as small decorations and excluded from layout manipulation.
- All remaining mesh objects are classified as movable furniture, with semantic sub-categories (sofa, chair, table, bed, rug, storage, decoration, etc.) inferred from name patterns.

The total number of movable objects is capped at `max_movable_objects` (default: 20) to prevent the JSON representation from exceeding the model's context window.

Room boundaries (`RoomBounds`) are computed automatically from the structural objects. If a single large mesh named "room" is found, its bounding box is used directly. Otherwise, the union of all structural object bounding boxes is computed.

The extracted state is saved to `scene_original.json`.

### Step 2 — Reference Renders

Two renders of the original scene are produced:
- A **top-down orthographic view**, useful for evaluating the 2D floor plan.
- An **isometric perspective view** at a configurable elevation and azimuth angle, used for visual quality assessment.

Both views are rendered at preview quality (512x512, 64 Cycles samples) for speed.

### Step 3 — Controlled Randomization

The `SceneRandomizer` produces a deep copy of the original state and displaces each movable object to a random position within the room bounds. Specifically:

- The X and Y coordinates are drawn uniformly from the valid range, which accounts for the object's own bounding box dimensions and a configurable wall margin (default: 10 cm), ensuring the object does not clip through walls.
- The Z coordinate is preserved unchanged to keep objects on the floor.
- Only the Z (yaw) rotation axis is randomized (configurable). This keeps furniture upright and physically plausible.
- An Axis-Aligned Bounding Box (AABB) overlap check is performed after each placement. If the new position causes excessive overlap (above `max_overlap_ratio`, default: 50% of the smaller object's footprint) with already-placed objects, a new position is sampled. Up to `max_placement_attempts` (default: 10) are made before accepting the best available position.

The randomized state is applied to the Blender scene via `SceneApplicator` and saved to `scene_randomized.json`. Two more renders are produced.

### Step 4 — LLM Text-Based Reorganization

The `SceneReorganizer` sends the disorganized scene state to Gemini as a structured JSON prompt. The prompt includes:

- A **system prompt** (`reorder_system.txt`) instructing the model to act as an expert interior designer, with explicit principles covering functional grouping, traffic flow (minimum 0.8 m pathways), focal point orientation, wall alignment for large furniture, visual balance, and overlap avoidance.
- A **user prompt** (`reorder_user.txt`) containing the room dimensions, the coordinate system specification (right-handed, +X East, +Y North, +Z Up, rotations in radians), and the JSON description of all movable and non-movable objects in their current disorganized state.

The model is instructed to return a valid JSON object with the same structure as the input, modifying only the `location` and `rotation_euler` fields. The Z coordinate and non-movable objects must remain unchanged.

The response is parsed using a three-strategy extraction approach (direct parse, markdown code block extraction, greedy JSON search). The parsed output is then validated and sanitized:

- Objects absent from the response fall back to their original positions.
- Coordinates that are not finite floats are rejected with fallback to the original.
- All coordinates are clamped to the room bounds.
- The Z coordinate of every movable object is always restored from the original state.

The validated state is saved to `scene_reordered.json`. Two more renders are produced.

### Step 5 — LLM Vision Critique and Refinement

The `VisualCritic` uploads the isometric render of the reordered scene to Gemini and requests a structured visual assessment. The prompt (`critic_user.txt`) asks the model to:

- Assign a quality score from 1 to 10.
- Identify specific objects that are misaligned, floating, overlapping, unnaturally rotated, or too close to walls.
- Suggest corrective actions (`move`, `rotate`, or `move_and_rotate`) with new coordinates.

If the model returns no corrections, the reordered state is promoted to the refined state unchanged. If corrections are present, up to `max_corrections` (default: 5) are applied. The same coordinate validation rules from Step 4 apply. The Z coordinate is always preserved.

The refined state is saved to `scene_refined.json`.

### Step 6 — Final High-Quality Render

A final render of the refined scene is produced at presentation quality (1280x720, 256 Cycles samples). Both top-down and isometric views are generated.

---

## Project Structure

```
NL2Scene3D-main/
├── config/
│   ├── prompts/
│   │   ├── critic_user.txt          # Prompt template for the vision critique step
│   │   ├── reorder_system.txt       # System prompt for the text-based reorganization
│   │   └── reorder_user.txt         # User prompt template for the reorganization step
│   └── settings.toml                # Application configuration (all non-secret parameters)
├── scenes/
│   └── originals/                   # Place .blend input files here
├── scripts/
│   ├── batch_pipeline.py            # Batch execution over multiple scenes
│   ├── run_pipeline.py              # Single-scene pipeline orchestrator
│   └── test_gemini.py               # Minimal API connectivity test
├── src/
│   └── nl2scene3d/
│       ├── __init__.py
│       ├── blender/
│       │   ├── __init__.py
│       │   ├── camera_setup.py      # Top-down and isometric camera configuration
│       │   └── renderer.py          # Automated rendering for each pipeline step
│       ├── config.py                # Centralized configuration loading (singleton)
│       ├── gemini_client.py         # Google Gemini API client with retry and fallback
│       ├── logging_setup.py         # Unified logging configuration
│       ├── metrics.py               # Quantitative evaluation of layout improvement
│       ├── models.py                # Shared dataclasses (SceneState, SceneObject, etc.)
│       ├── randomizer.py            # Controlled scene disorganization
│       ├── scene_applicator.py      # Applies SceneState transformations to Blender objects
│       ├── scene_loader.py          # Scene introspection, classification, and JSON I/O
│       ├── scene_reorganizer.py     # LLM text call for layout reorganization
│       └── visual_critic.py         # LLM vision call for layout critique and refinement
├── tests/                           # Unit tests (require no Blender installation)
├── .env.example                     # Template for the required environment variables
├── pyproject.toml                   # Build system and project metadata
├── requirements.txt                 # Runtime dependencies
└── LICENSE                          # MIT License
```

---

## Module Reference

### `models.py`

Defines the core data structures shared across the entire pipeline.

| Class | Description |
|---|---|
| `ObjectTransform` | Holds `location`, `rotation_euler`, and `dimensions` as lists of three floats. Validates component count on construction. |
| `SceneObject` | Represents a single Blender object with its name, type, transform, semantic category, and `is_movable` flag. |
| `RoomBounds` | Encodes the spatial extents of the room (X/Y min-max, floor and ceiling Z). Provides a `clamp_location` utility. |
| `SceneState` | The complete state of a scene at a given pipeline step: list of objects, room bounds, step label, and metadata dictionary. Provides `movable_objects`, `static_objects`, and `get_object_by_name` accessors. Supports deep copy and JSON round-trip serialization. |
| `LLMCorrection` | Represents a single corrective suggestion from the vision model: target object name, action type, optional new location and rotation, and textual reasoning. |

### `config.py`

Implements a thread-safe singleton configuration loader. Reads `config/settings.toml` and merges values with environment variables. The priority chain is: environment variable > TOML value > hardcoded default.

| Dataclass | Contents |
|---|---|
| `GeminiConfig` | API key, primary and fallback model names, retry count, timeout, temperature, max output tokens. |
| `RenderConfig` | Preview and final render dimensions and sample counts. Camera parameters for both views. |
| `RandomizerConfig` | Seed, jitter ratio, Z-only rotation flag, overlap checking, wall margin, overlap threshold, max placement attempts. |
| `PipelineConfig` | Scene and output directories, movable object limit, object classification patterns. |
| `LoggingConfig` | Log level, format, optional file output. |
| `AppConfig` | Aggregates all of the above. Loaded via `get_config()`. |

`reset_config()` is provided exclusively for use in unit tests to reset the singleton between test cases.

### `gemini_client.py`

Wraps the `google-generativeai` SDK. Provides two public methods:

- `call_text(system_prompt, user_prompt)`: Sends a text-only request. Used by the reorganizer.
- `call_vision(image_path, user_prompt)`: Uploads the render image via `genai.upload_file` and sends a multimodal request. The uploaded file is deleted from Google's servers in a `finally` block regardless of the outcome.

Both methods implement retry with exponential backoff (`2^(attempt+1)` seconds for rate limit errors, `2^attempt` for other API errors). If the primary model hits a persistent rate limit, the call is transparently retried using the fallback model. JSON extraction from the response applies three strategies in sequence: direct parse, markdown code block extraction, and greedy brace/bracket matching.

### `scene_loader.py`

Handles all interaction with the Blender object database. `extract_scene_state()` iterates over `bpy.context.scene.objects`, reads location, rotation, and bounding box dimensions for each object, and classifies it using `_classify_object()`. Room bounds are computed by `extract_room_bounds_from_objects()` using the strategy described in the pipeline section. `save_state_to_json()` and `load_state_from_json()` provide JSON persistence.

### `randomizer.py`

The `SceneRandomizer` class takes a `SceneState` and returns a new one with movable objects displaced. Static objects are copied unchanged. The randomization loop attempts to place each object without excessive AABB overlap. If no valid placement is found within the attempt limit, the last computed position is used and a warning is logged.

### `scene_reorganizer.py`

Builds the text prompt from templates, calls `GeminiClient.call_text()`, and passes the raw response through `_validate_and_sanitize_llm_output()`. If parsing fails entirely, the disorganized state is returned with `pipeline_step="reordered_failed"` and the error recorded in metadata.

### `scene_applicator.py`

`SceneApplicator.apply_state()` applies a `SceneState` to the live Blender scene by matching objects by name and updating `location` and `rotation_euler` properties. A configurable `tolerance` (default: 1 mm) prevents redundant updates for nearly-identical values. After all updates, `bpy.context.view_layer.update()` is called to ensure Blender's internal state is consistent before rendering.

### `blender/camera_setup.py`

Configures two dedicated camera objects prefixed with `NL2Scene3D_Camera_`. The top-down camera uses orthographic projection scaled to cover the room footprint with a small padding margin. The isometric camera uses perspective projection positioned at the configured elevation and azimuth angles relative to the room center, at a distance proportional to the room's horizontal extent.

### `blender/renderer.py`

`BlenderRenderer.render_step()` configures the Cycles render engine, positions the appropriate camera, executes the render, and saves the output as a PNG file. Preview renders use reduced resolution and sample counts; the final render uses the higher-quality settings.

### `visual_critic.py`

`VisualCritic.critique_and_refine()` sends the isometric render to `GeminiClient.call_vision()`, parses the structured response into a list of `LLMCorrection` objects, and applies them via `_apply_corrections_to_state()`. Corrections are applied regardless of whether the score is above or below the quality threshold: the threshold controls a log message only, not whether corrections are applied.

### `metrics.py`

`compute_metrics()` compares a evaluated state against the original (ground truth) state for all movable objects that appear in both. It computes:

- **Mean 2D Euclidean position delta** (meters): how far, on average, the objects are from their original positions.
- **Mean angular difference on the Z axis** (radians): normalized to [0, pi].
- **Improvement score** [0.0, 1.0]: `1 - (mean_reordered_delta / mean_randomized_delta)`. A score of 1.0 means the reordered layout is identical to the original; 0.0 means no improvement over the random layout.

`compute_pipeline_metrics()` computes all three states (randomized, reordered, refined) in a single call.

---

## Configuration

The file `config/settings.toml` contains all configurable parameters. The following sections are defined:

| Section | Key parameters |
|---|---|
| `[gemini]` | `model_primary`, `model_fallback`, `max_retries`, `timeout_seconds`, `temperature`, `max_output_tokens` |
| `[pipeline]` | `max_movable_objects`, `min_object_dimension_meters`, `wall_margin_meters`, `max_overlap_ratio`, `max_placement_attempts`, `min_quality_score_for_corrections`, `max_corrections_to_apply` |
| `[randomizer]` | `seed` (0 = clock-based), `rotate_z_only`, `check_overlaps` |
| `[render.preview]` | `width`, `height`, `samples`, `engine` |
| `[render.final]` | `width`, `height`, `samples`, `engine` |
| `[render.camera]` | `isometric_elevation_degrees`, `isometric_azimuth_degrees`, `isometric_focal_length_mm`, `isometric_distance_multiplier`, `topdown_height_multiplier`, `topdown_ortho_scale_padding` |
| `[paths]` | `scenes_dir`, `outputs_dir`, `prompts_dir` |
| `[object_classification]` | `non_mesh_types`, `structural_name_patterns`, `ceiling_light_patterns` |
| `[logging]` | `level`, `format`, `datefmt`, `write_to_file`, `log_file` |
| `[metrics]` | `enabled`, `min_improvement_distance_meters` |

Any of these values can be overridden by setting the corresponding environment variable (see `.env.example` for the full list of recognized variable names).

---

## Requirements

- **Python** >= 3.10
- **Blender** >= 4.0 (for pipeline execution; not required for unit tests)
- **Google Gemini API key** (free tier is sufficient for moderate usage; the default primary model is `gemini-3-flash-preview` with a free-tier limit of 500 requests per day)

Python dependencies:

```
google-generativeai>=0.8.0
google-api-core>=2.15.0
python-dotenv>=1.0.0
tomli>=2.0.0  # only for Python < 3.11
```

Development dependencies (linting, formatting, testing):

```
black>=24.0.0
ruff>=0.4.0
pytest>=8.0.0
pytest-cov>=5.0.0
```

---

## Installation

**1. Clone the repository.**

```bash
git clone <repository-url>
cd NL2Scene3D-main
```

**2. Create and activate a virtual environment.**

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

**3. Install the package and its dependencies.**

```bash
pip install -e ".[dev]"
```

**4. Configure environment variables.**

```bash
cp .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` to your Google Gemini API key. All other variables in `.env.example` are optional overrides for settings already present in `settings.toml`.

**5. Place input scenes.**

Copy your Blender `.blend` files into `scenes/originals/`.

---

## Usage

### Verify API connectivity

Before running the full pipeline, confirm that the API key is valid and the model responds correctly:

```bash
python scripts/test_gemini.py
```

A successful run logs the message `Test superato. API Gemini funzionante.`

### Run the pipeline on a single scene

The pipeline must be executed through Blender's embedded Python interpreter. Blender loads the specified `.blend` file and then runs the pipeline script. Arguments after `--` are passed to the Python script.

```bash
blender --background scenes/originals/living_room.blend \
    --python scripts/run_pipeline.py -- \
    --scene-name living_room \
    --output-dir scenes/outputs/living_room \
    --seed 42 \
    --log-level INFO
```

Available arguments:

| Argument | Required | Description |
|---|---|---|
| `--scene-name` | Yes | Identifier used in output file names |
| `--output-dir` | Yes | Directory where all output files are written |
| `--prompts-dir` | No | Override for the prompt templates directory |
| `--seed` | No | Integer seed for the randomizer (overrides config) |
| `--skip-vision` | No | Flag: skips the vision critique step |
| `--max-objects` | No | Override for the movable object count limit |
| `--log-level` | No | One of: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Run the pipeline on all scenes in a directory

```bash
blender --background \
    --python scripts/batch_pipeline.py -- \
    --scenes-dir scenes/originals \
    --outputs-dir scenes/outputs \
    --seed 42 \
    --log-level INFO
```

The batch runner iterates over all `.blend` files in the specified directory (configurable via `--scene-pattern`), processes each one sequentially, inserts a 2-second pause between scenes to respect API rate limits, and writes a summary report to `scenes/outputs/batch_report.json`.

---

## Output Artifacts

For each processed scene, the following files are written to the output directory:

| File | Description |
|---|---|
| `scene_original.json` | Full scene state as extracted from the `.blend` file |
| `scene_randomized.json` | Scene state after artificial disorganization |
| `scene_reordered.json` | Scene state after the first LLM reorganization pass |
| `scene_refined.json` | Scene state after the vision critique and correction pass |
| `render_original_top.png` | Top-down orthographic render of the original scene |
| `render_original_iso.png` | Isometric perspective render of the original scene |
| `render_randomized_top.png` | Top-down render of the randomized scene |
| `render_randomized_iso.png` | Isometric render of the randomized scene |
| `render_reordered_top.png` | Top-down render after LLM text reorganization |
| `render_reordered_iso.png` | Isometric render after LLM text reorganization (used as vision input) |
| `render_final_top.png` | Final high-quality top-down render |
| `render_final_iso.png` | Final high-quality isometric render |

For batch runs, an additional file is written to the root output directory:

| File | Description |
|---|---|
| `batch_report.json` | Summary of all processed scenes: status, duration, and error messages |

---

## Metrics and Evaluation

When metrics are enabled (`[metrics] enabled = true` in `settings.toml`), the pipeline computes quantitative indicators for the reordered and refined states relative to the original layout.

The **mean position delta** measures, in meters, the average 2D Euclidean distance between each movable object's final position and its original position. A lower value indicates a layout closer to the original arrangement.

The **mean rotation delta** measures, in radians, the average angular difference on the Z axis between the final and original rotations, normalized to the range [0, pi].

The **improvement score** normalizes the position delta against the randomized baseline. A value of 1.0 indicates that the pipeline has exactly recovered the original layout. A value of 0.0 indicates that the reordered layout is no closer to the original than the randomized one. Negative values are clamped to 0.0.

These metrics are logged at the INFO level at the end of each pipeline run and, if the output is serialized to JSON, included in the metadata field of the refined `SceneState`.

---

## Development and Testing

### Run the unit test suite

The tests do not require Blender. They use mock implementations of `bpy` to test all Blender-dependent modules in isolation.

```bash
pytest tests/ -v --cov=src/nl2scene3d --cov-report=term-missing
```

### Code formatting and linting

```bash
black src/ tests/ scripts/
ruff check src/ tests/ scripts/ --fix
```

The project targets Python 3.10 compatibility. Line length is set to 88 characters. The Ruff linter enforces pycodestyle, pyflakes, isort, flake8-bugbear, flake8-comprehensions, pyupgrade, PEP 8 naming, and type annotation rules, with selective exemptions documented in `pyproject.toml`.

### Note on Blender's Python environment

When the pipeline is executed through Blender, the active Python interpreter is Blender's own embedded interpreter, which is isolated from the system Python and the project's virtual environment. The `run_pipeline.py` script addresses this by injecting the project's `src/` directory and the virtual environment's `site-packages/` directory into `sys.path` at startup, making the `nl2scene3d` package and its dependencies available to Blender's interpreter.

---

## License

This project is released under the MIT License. See the `LICENSE` file for the full text.
