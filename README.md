
# NL2Scene3D

Scene Reorganization from Random to Ordered via Multimodal Language Models

## Panoramica

NL2Scene3D e' una pipeline che prende scene 3D gia' arredate, le disorganizza
artificialmente e usa un MLLM (Gemini) per riordinarle, raffinando il risultato
tramite feedback visivo.

## Requisiti

- Python >= 3.10
- Blender >= 4.0 (per l'esecuzione della pipeline)
- Chiave API Google Gemini (free tier sufficiente)

## Setup

1. Clonare il repository
2. Creare l'ambiente virtuale:
   ```
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
3. Installare le dipendenze:
   ```
   pip install -e ".[dev]"
   ```
4. Configurare le variabili d'ambiente:
   ```
   cp .env.example .env
   # Inserire GEMINI_API_KEY nel file .env
   ```

## Esecuzione della pipeline

```bash
blender --background scenes/originals/my_scene.blend \
    --python scripts/run_pipeline.py -- \
    --scene-name my_scene \
    --output-dir scenes/outputs/my_scene
```

## Esecuzione dei test

```bash
pytest tests/ -v
```

## Verifica API

```bash
python scripts/test_gemini.py
```

## Struttura degli output

Per ogni scena processata vengono prodotti:
- `scene_original.json`: Stato iniziale estratto dal file .blend
- `scene_randomized.json`: Stato dopo la disorganizzazione artificiale
- `scene_reordered.json`: Output della prima chiamata LLM
- `scene_refined.json`: Output della seconda chiamata LLM Vision
- `render_original_top.png`: Vista top-down della scena originale
- `render_original_iso.png`: Vista isometrica della scena originale
- `render_randomized_top.png`, `render_randomized_iso.png`
- `render_reordered_top.png`, `render_reordered_iso.png`
- `render_final_top.png`, `render_final_iso.png`

## Flusso di lavoro

Il sistema esegue le seguenti operazioni:

- Caricamento del file .blend
- Estrazione JSON degli oggetti e delle loro trasformazioni
- Render di riferimento (top-down + isometrico)
- Randomizzazione controllata di posizioni e rotazioni
- Render della scena disorganizzata
- Estrazione JSON dello stato disorganizzato
- Chiamata LLM: riordino testuale basato su JSON
- Applicazione delle coordinate suggerite e rendering
- Chiamata LLM Vision: critica del render isometrico
- Applicazione delle rifiniture e render finale ad alta qualita'

## Nota finale sull'esecuzione

Il progetto e' ora completo nelle sue componenti principali. Di seguito le istruzioni operative:

**Test unitari** (eseguibili senza Blender):
```bash
cd NL2Scene3D
pip install -e ".[dev]"
pytest tests/ -v --cov=src/nl2scene3d
```

**Verifica API Gemini**:
```bash
python scripts/test_gemini.py
```

**Esecuzione pipeline completa** (richiede Blender):
```bash
blender --background scenes/originals/living_room.blend \
    --python scripts/run_pipeline.py -- \
    --scene-name living_room \
    --output-dir scenes/outputs/living_room \
    --seed 42 \
    --log-level INFO
```

**Linting e formatting**:
```bash
black src/ tests/ scripts/
ruff check src/ tests/ scripts/ --fix
```