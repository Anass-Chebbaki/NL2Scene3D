# NL2Scene3D

**Scene Reorganization from Random to Ordered via Multimodal Language Models**

Progetto universitario di Computer Graphics — Aprile 2026

## Descrizione

NL2Scene3D prende scene 3D già arredate di qualità, ne disorganizza randomicamente
gli oggetti, e usa un MLLM (Gemini 2.5 Flash) per riordinarle in modo sensato,
raffinando il risultato tramite feedback visivo.

## Pipeline

1. Carica scena `.blend` già arredata
2. Estrai JSON con asset + coordinate originali (`bpy`)
3. Render originale (top-down + isometrico)
4. Randomizza posizione + rotazione degli oggetti
5. Render scena disordinata
6. Estrai JSON della scena disordinata
7. **1ª chiamata LLM (testo):** riordina la scena
8. Applica coordinate + render riordinata
9. **2ª chiamata LLM (Vision):** raffina il render
10. Applica rifiniture + render finale

## Tecnologie

- Python 3.13
- Blender 5.1 + `bpy`
- Gemini 2.5 Flash (google-genai)

## Setup

```bash
# Clona la repository
git clone https://github.com/TUO_USERNAME/NL2Scene3D.git
cd NL2Scene3D

# Crea ambiente virtuale
python -m venv .venv
.venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt

# Configura API key
copy .env.example .env
# → apri .env e inserisci la tua GEMINI_API_KEY
```

## Struttura