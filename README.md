# NL2Scene3D

Add-on per Blender che riorganizza scene 3D esistenti tramite un modello linguistico di grandi dimensioni (LLM). L'add-on genera un prompt strutturato e una descrizione JSON della scena, ottiene dal modello una lista di posizioni e rotazioni, e la applica garantendo il rispetto di tutti i vincoli geometrici — indipendentemente dalla qualità dell'output del modello.

Sono disponibili due flussi di lavoro, intercambiabili e che condividono la stessa pipeline di sanitizzazione:

- **Flusso manuale (human-in-the-loop)**. L'add-on copia il prompt negli appunti e produce i render etichettati; l'utente li sottopone all'LLM di propria scelta (API, interfaccia web, strumenti locali) e incolla la risposta nell'add-on.
- **Flusso automatico via API**. L'add-on contatta direttamente un provider LLM (Google Gemini, Anthropic o OpenAI), allega i render e applica la risposta, il tutto con un solo click. Richiede una API key configurata nelle preferenze.

---

## Indice

- [Panoramica del progetto](#panoramica-del-progetto)
- [Principi di progettazione](#principi-di-progettazione)
- [Architettura](#architettura)
  - [Struttura delle directory](#struttura-delle-directory)
  - [Layer Blender](#layer-blender)
  - [Core puro Python](#core-puro-python)
  - [Flusso dati](#flusso-dati)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
  - [Installazione di Pillow](#installazione-di-pillow)
- [Utilizzo passo per passo](#utilizzo-passo-per-passo)
  - [Preparazione della scena](#preparazione-della-scena)
  - [Scala a misura reale](#scala-a-misura-reale)
  - [Ispezione e classificazione (dry-run)](#ispezione-e-classificazione-dry-run)
  - [Override manuali](#override-manuali)
  - [Step 1 — Randomize Layout](#step-1--randomize-layout)
  - [Step 2a — Render con etichette](#step-2a--render-con-etichette)
  - [Istruzioni personalizzate per l'LLM](#istruzioni-personalizzate-per-lllm)
  - [Riordino automatico via API](#riordino-automatico-via-api)
  - [Step 2b — Esporta prompt per LLM](#step-2b--esporta-prompt-per-llm)
  - [Step 2c — Applica la risposta dell'LLM](#step-2c--applica-la-risposta-dellllm)
  - [Reset allo stato originale](#reset-allo-stato-originale)
  - [Metriche di spostamento](#metriche-di-spostamento)
- [Classificazione automatica degli oggetti](#classificazione-automatica-degli-oggetti)
- [Calcolo dei confini della stanza](#calcolo-dei-confini-della-stanza)
- [Sistema di grouping padre-figlio](#sistema-di-grouping-padre-figlio)
- [Algoritmo di randomizzazione](#algoritmo-di-randomizzazione)
- [Pipeline di render](#pipeline-di-render)
- [Costruzione del prompt e del payload JSON](#costruzione-del-prompt-e-del-payload-json)
- [Sanitizzazione della risposta LLM](#sanitizzazione-della-risposta-llm)
- [Garanzie geometriche](#garanzie-geometriche)
- [Configurazione](#configurazione)
- [Testing offline](#testing-offline)
- [Struttura del progetto](#struttura-del-progetto)
- [Licenza](#licenza)

---

## Panoramica del progetto

NL2Scene3D affronta il problema della riorganizzazione automatica di layout di arredamento 3D in Blender. A differenza di approcci che generano scene da zero, il progetto opera su scene preesistenti: importa la geometria già modellata, la disordina in modo controllato per eliminare qualsiasi bias di posizionamento originale, e poi chiede a un LLM di proporre un layout funzionale e realistico a partire da una descrizione strutturata.

Il modello non accede direttamente alla scena e non conosce la geometria 3D: riceve esclusivamente un payload JSON con le impronte XY degli oggetti, i confini della stanza e le relazioni di gruppo, più i render etichettati come riferimento visivo. La risposta del modello — una lista di posizioni XY e rotazioni — viene poi validata e applicata da un layer di sanitizzazione deterministica che garantisce la correttezza geometrica indipendentemente da cosa il modello abbia proposto.

L'add-on offre due modalità per interagire con il modello. Nel flusso manuale non viene effettuata alcuna chiamata di rete: l'utente porta autonomamente prompt e immagini all'LLM di propria scelta e incolla o carica la risposta nell'add-on. Nel flusso automatico l'add-on contatta direttamente il provider (Gemini, Anthropic o OpenAI) tramite il modulo `llm_providers.py`, che è l'unico punto del progetto ad effettuare chiamate di rete ed è tenuto deliberatamente fuori da `core/`. In entrambi i casi il package `core/` resta puro e offline: non importa `bpy` e non apre connessioni.

---

## Principi di progettazione

**Invarianza della coordinata Z.** Nessuna operazione dell'add-on — né il randomizzatore né la sanitizzazione della risposta LLM — modifica mai la componente Z della posizione di un oggetto. Un oggetto che si trova a 0.80 m di altezza (ad esempio poggiato su una scrivania) mantiene quella quota attraverso tutte le fasi. Non esiste alcuna funzione "drop to floor". Questo principio è applicato a livello architetturale: tutti i moduli che calcolano posizioni lavorano solo nel piano XY.

**Correttezza geometrica garantita dal codice, non dal modello.** Il modello propone posizioni; il sanitizzatore le corregge. Il risultato finale è sempre geometricamente valido: nessun oggetto fuori dai muri, nessuna sovrapposizione tra mobili, nessun oggetto che blocca porte o finestre. Se il modello produce un layout impossibile, il sistema lo corregge; se produce un layout eccellente, lo applica senza modifiche.

**Flusso human-in-the-loop.** L'utente mantiene il controllo completo: sceglie quale LLM usare, può ispezionare il prompt prima di inviarlo, verifica il risultato dopo l'applicazione e può annullare con un click.

**Separazione netta tra logica e interfaccia Blender.** Tutto il codice geometrico, di collision detection, di randomizzazione e di reorganizzazione vive in `core/`, un package Python puro senza dipendenze da `bpy`. Gli operatori Blender in `operators.py` sono deliberatamente sottili: orchestrano il core e gestiscono la UI, senza contenere logica geometrica diretta. Le chiamate di rete verso i provider LLM sono a loro volta isolate in un modulo dedicato (`llm_providers.py`), separato dal core. Tutti questi livelli sono esercitabili senza Blender installato.

**Nessuna categoria semantica hard-coded per i mobili.** L'add-on non mantiene liste di parole chiave per "letto", "sedia", "tavolo" e simili: sarebbero fragili e non generalizzerebbero tra scene diverse. La classificazione automatica distingue solo tre categorie: `technical` (camera e luci), `structural` (elementi identificati per nome: muri, pavimento, soffitto, porte, finestre) e `object` (tutto il resto). La distinzione fisso/mobile per gli oggetti è decisa dall'utente tramite il pannello override.

---

## Architettura

### Struttura delle directory

```
nl2scene3d_addon/
├── __init__.py          Entry point dell'add-on Blender (register / unregister)
├── operators.py         Operatori Blender — orchestrazione pura, nessuna logica geometrica
├── ui.py                Pannello sidebar, UIList override, preferenze add-on
├── llm_providers.py     Client di rete per i provider LLM (Gemini/Anthropic/OpenAI) 
└── core/                Package Python puro — nessuna dipendenza da bpy
    ├── __init__.py      Documentazione del package
    ├── models.py        Dataclass: Transform, SceneObject, RoomBounds, SceneState
    ├── geometry.py      SAT, MTV, AABB/OBB, collision score, clearance porte/finestre
    ├── classify.py      Classificazione oggetti, calcolo RoomBounds, parenting manuale
    ├── randomizer.py    SceneRandomizer: disordine controllato con evitamento collisioni
    ├── reorganizer.py   Builder prompt/payload, estrattore JSON, sanitizzatore risposta
    ├── render.py        Pipeline render: etichette gutter, bussola assi, barra di scala
    ├── scene_io.py      Bridge Blender ↔ core: extract_scene_state, apply_state, metriche
    └── settings.py      Constants (frozen dataclass), NON_MESH_TYPES, STRUCTURAL_PATTERNS

offline_testing/
├── PROMPT+JSONSCENE.txt   Esempio completo di prompt + JSON per una camera da letto
└── PYTHON - SCRIPT.txt    Script Blender minimale per applicare un JSON direttamente
```

### Layer Blender

`operators.py` contiene tutti gli operatori Blender. Ogni operatore segue lo stesso schema: chiama `scene_io.extract_scene_state()` per ottenere un `SceneState` dalla scena corrente, invoca il modulo core appropriato, poi chiama `scene_io.apply_state()` per riscrivere le pose in Blender. Gli operatori gestiscono la progress bar, il cursore di attesa, il reporting degli errori e la scrittura nei Text datablock; non eseguono mai calcoli geometrici direttamente.

`ui.py` registra le classi UI di Blender: `NL2SCENE3D_AddonPreferences` (seed del randomizer più tutte le preferenze per la chiamata API automatica: provider, API key per provider, modello, temperature, timeout, numero di retry e toggle auto-render), `NL2_ObjectOverride` (PropertyGroup per ogni voce della lista override), `NL2SCENE3D_UL_overrides` (UIList con toggle etichetta/fisso/padre/keep_scale) e `NL2SCENE3D_PT_main_panel` (il pannello principale nella sidebar della 3D View). Il modulo viene importato solo all'interno di `register()` per garantire che i moduli puri del core restino importabili senza Blender.

`llm_providers.py` è l'unico modulo dell'add-on che effettua chiamate di rete, ed è tenuto fuori da `core/` per non violarne la purezza. Usa solo la libreria standard (`urllib`, `ssl`, `base64`, `json`), quindi non richiede dipendenze esterne nel Python di Blender e resta importabile senza Blender. Espone `call_llm()` come dispatcher unico verso `call_gemini()`, `call_anthropic()` e `call_openai()`, più l'helper `_http_post_json()` che gestisce i retry con backoff esponenziale sugli errori transitori (HTTP 429/500/502/503/504 e blip di rete), rispettando l'header `Retry-After`. Il modulo prende un prompt e una lista di immagini e restituisce il testo grezzo della risposta, che viene poi passato a `reorganizer.extract_json()` esattamente come nel flusso copia-incolla. La logica geometrica resta interamente in `core/`.

### Core puro Python

`models.py` definisce il contratto dati dell'intero sistema. `Transform` contiene location, rotation_euler, dimensions e origin_offset (offset dell'origine rispetto al centro geometrico della mesh, già moltiplicato per la scala dell'oggetto) e offre i metodi geometrici `geometric_center_xy()`, `aabb_xy()`, `obb_corners_xy()` e `z_range()`. `SceneObject` aggrega un `Transform` con metadati (nome, tipo Blender, categoria, is_movable, parent, children). `RoomBounds` descrive il rettangolo di pavimento della stanza con metodi di contenimento e clamping. `SceneState` è lo snapshot completo della scena a un determinato passo della pipeline, con una cache interna nome→oggetto ricostruita automaticamente.

`geometry.py` è l'unica fonte di verità per la geometria di collisione. Implementa il Separating Axis Theorem (`sat_overlap`) su poligoni 2D convessi, `wall_collision()` per le collisioni con i muri fisici (escludendo porte, finestre e mesh-stanza), `furniture_collision()` per le collisioni tra mobili, `check_openings_clearance()` per la zona di rispetto davanti ad aperture, `has_collision()` come entry point principale, `collision_score()` come misura scalare della "bruttezza" di una posizione (usato dal randomizer per scegliere il fallback migliore), `penetration_vector()` per il Minimum Translation Vector e `group_aabb_xy()` per l'AABB combinato di un gruppo padre+figli a una posizione proposta.

`classify.py` gestisce la classificazione degli oggetti (`default_classification`, `resolve_classification`), il calcolo dei confini della stanza (`compute_room_bounds`) e l'assegnazione dei rapporti padre-figlio manuali (`apply_manual_parents`). Contiene anche `suggest_grouping()`, che propone relazioni padre-figlio basate esclusivamente sulla geometria (SAT + prossimità verticale + rapporto di dimensioni).

`randomizer.py` implementa `SceneRandomizer`, la classe che disordina il layout. Gestisce la trasformazione rigida di gruppo (`apply_rigid_transform`), il clamping del gruppo dentro i confini (`_clamp_parent_group_location`), la generazione di posizioni casuali valide (`_random_location`) e rotazioni casuali a multipli di 90° (`_random_rotation`).

`reorganizer.py` è diviso in tre responsabilità: costruzione del payload JSON e del prompt (`build_request`, `build_prompt`, `PROMPT_TEMPLATE`), estrazione robusta del JSON dalla risposta del modello (`extract_json`), e sanitizzazione e applicazione della risposta (`sanitize_response`).

`render.py` gestisce la pipeline di render. Le funzioni pure (utilizzabili senza Blender) includono la conversione NDC→pixel, il declutter delle etichette per sovrapposizione (MTV in spazio pixel), il layout a gutter, il calcolo della lunghezza "tonda" per la barra di scala e il disegno via Pillow. Le funzioni che richiedono Blender creano e rimuovono camera temporanee (top-down, corner prospettica, isometrica) e eseguono il render OpenGL con fallback al renderer standard.

`scene_io.py` è l'unico modulo del core con dipendenza da `bpy`. Gestisce la lettura della scena (`extract_scene_state`), la scrittura delle pose (`apply_state`), il salvataggio e ripristino dello stato originale (custom property `nl2_home_loc` / `nl2_home_rot`), la cattura di snapshot di posa per le metriche (`capture_pose_snapshot`) e la generazione del report di spostamento (`format_metrics_report`, `build_metrics_report`).

`settings.py` definisce `Constants` come dataclass frozen con tutti i parametri operativi, più le costanti `NON_MESH_TYPES` e `STRUCTURAL_PATTERNS`. L'istanza globale `CONST` è usata come default in tutto il package; i test possono istanziare `Constants` con valori diversi senza effetti collaterali.

### Flusso dati

```
Scena Blender
     |
     | scene_io.extract_scene_state()
     v
SceneState (pipeline_step="original")
     |
     | SceneRandomizer.randomize()
     v
SceneState (pipeline_step="randomized")
     |
     | scene_io.apply_state()          render.render_labeled_views()
     v                                          |
Scena Blender disordinata              PNG etichettati
     |                                          |
     | scene_io.extract_scene_state()           |
     v                                          |
SceneState + reorganizer.build_prompt() --------+
     |
     | (utente copia prompt+immagini nel proprio LLM)
     v
Risposta JSON dell'LLM
     |
     | reorganizer.sanitize_response()
     v
SceneState (pipeline_step="reorganized")
     |
     | scene_io.apply_state()
     v
Scena Blender riorganizzata
```

Il diagramma illustra il flusso manuale. Nel flusso automatico i tre passaggi centrali (copia del prompt → LLM → incolla della risposta) sono sostituiti da un'unica chiamata interna a `llm_providers.call_llm()`; il resto della pipeline è identico.

---

## Requisiti

| Dipendenza | Versione | Note |
|---|---|---|
| Blender | 4.2 o superiore | Compatibile con 4.2 LTS e 5.1.x; la versione minima è conservativa e può essere alzata se si usano API più recenti |
| Python | 3.10 o superiore | Incluso nella distribuzione di Blender |
| Pillow | qualsiasi versione recente | **Opzionale.** Necessario per il disegno delle etichette sui render e la barra di scala. Se assente, le coordinate delle etichette vengono scritte in un file `.labels.json` accanto all'immagine e la funzione `_draw_labels()` restituisce `False` |

Il flusso automatico via API non richiede alcuna dipendenza aggiuntiva: `llm_providers.py` usa solo `urllib/ssl` (già inclusi in Blender), quindi le chiamate HTTPS funzionano out-of-the-box, senza installare requests o gli SDK ufficiali dei provider. È invece necessaria una API key valida per il provider scelto (Gemini, Anthropic o OpenAI), configurata nelle preferenze dell'add-on o esposta come variabile d'ambiente. Il flusso manuale non richiede né API key né connessione di rete.

---

## Installazione

1. Clonare o scaricare il repository.

2. Creare il file zip dell'add-on. Puoi farlo automaticamente eseguendo lo script Python incluso nella root del progetto:
   ```bash
   python build_addon.py
   ```
   Questo genererà un file zip ottimizzato come `nl2scene3d_addon_v1.0.0.zip` escludendo cache e file inutili. In alternativa, puoi crearlo manualmente a partire dalla directory `nl2scene3d_addon/`:
   ```bash
   zip -r nl2scene3d_addon.zip nl2scene3d_addon/
   ```
   La directory `offline_testing/` non deve essere inclusa nello zip.

3. In Blender:
   - **Per Blender 4.2 o superiore**: Apri **Modifica > Preferenze > Get Extensions** (Ottieni estensioni), clicca sull'icona della freccia in alto a destra e seleziona **Install from Disk** (Installa da disco), quindi seleziona il file zip appena generato.
   - **Per Blender 4.1 o inferiore**: Apri **Modifica > Preferenze > Add-on > Installa**, seleziona lo zip e abilita **NL2Scene3D** dalla lista.

4. Il pannello dell'add-on compare nella **sidebar della 3D Viewport** (tasto N) sotto la tab **NL2Scene3D**.

### Installazione di Pillow

Pillow non è incluso nella distribuzione standard di Blender, ma è richiesto per scrivere le etichette dei nomi e disegnare la scala/bussola sui render PNG.

#### Metodo 1: Autoinstallazione ad un click (Consigliato)
Se Pillow non è installato, all'avvio l'add-on mostra un avviso rosso nella Sidebar di Blender con il pulsante **"Installa Pillow"**:
1. Clicca sul pulsante **Installa Pillow**.
2. L'add-on individuerà l'eseguibile Python corrente di Blender ed eseguirà `pip install Pillow --user` in background.
3. L'installazione avviene in spazio utente (senza richiedere diritti di amministratore) e la libreria viene caricata a runtime senza bisogno di riavviare Blender.

#### Metodo 2: Installazione manuale da terminale
Se preferisci farlo manualmente, apri il terminale del tuo sistema operativo ed esegui pip puntando al Python **interno** di Blender:

```bash
# Sostituire il percorso con quello della propria installazione di Blender.
# Su Linux / macOS:
/percorso/a/blender/5.1/python/bin/python3 -m pip install Pillow --user

# Su Windows:
"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" -m pip install Pillow --user
```
*(Sostituisci `5.1` con la tua versione di Blender)*

Se Pillow non è disponibile (o l'installazione fallisce), i render vengono comunque prodotti ma salvati come immagini "pulite" senza etichette; le coordinate delle etichette vengono invece salvate in un file `.labels.json` accanto all'immagine.

---

## Utilizzo passo per passo

### Preparazione della scena

Aprire un file `.blend` esistente contenente la scena da riorganizzare. L'add-on è agnostico rispetto al tipo di ambiente: camere da letto, uffici, cucine, spazi retail, sale conferenze e qualsiasi altro interno sono supportati senza configurazione specifica.

L'add-on opera sulle pose in **spazio mondo** lette da `matrix_world`, gestendo correttamente oggetti con parent nativi Blender, scale non uniformi e rotation mode diverse da XYZ (che vengono normalizzate a XYZ durante `extract_scene_state()`).

### Scala a misura reale

L'add-on lavora in **metri reali**. Se la scena è stata importata in centimetri, pollici o a scala arbitraria, è necessario correggerla prima di procedere. Premere **Scala a misura reale** nel pannello e scegliere una delle due modalità:

- **Oggetto noto:** selezionare un oggetto di cui si conosce la dimensione reale (ad esempio una porta standard di 2.10 m) e indicarne la misura. Il fattore di scala viene calcolato come `dimensione_reale / dimensione_corrente_massima` dell'oggetto selezionato.
- **Dimensione stanza:** indicare la lunghezza reale del lato maggiore della stanza. Il fattore viene calcolato rispetto all'AABB unione di tutti i mesh.

La scala viene applicata come trasformazione uniforme attorno al centro geometrico della scena (`T(centro) · Scale(f) · T(-centro)`), processando i padri prima dei figli per evitare la doppia applicazione. Se **Applica scala** è attivo (consigliato), la scala viene applicata ai mesh radice via `bpy.ops.object.transform_apply`. Al termine, lo stato originale salvato viene azzerato perché non sarebbe più coerente con la nuova scala.

### Ispezione e classificazione (dry-run)

Prima di eseguire qualsiasi operazione, premere **Inspect Scene** per verificare come l'add-on interpreta la scena corrente. Il report, scritto nel Text datablock `NL2_Inspect_Report` e nella console di sistema, mostra:

- I confini della stanza calcolati (X, Y, Z).
- Il conteggio totale di oggetti, mobili, fissi e gruppi.
- Una tabella con nome, categoria, stato fisso/mobile e padre per ogni oggetto.
- Avvisi per oggetti mobili con nomi strutturali (es. un muro classificato come mobile per errore).
- Avvisi per oggetti con un'impronta XY sospettamente grande rispetto alla stanza (possibile errore di scala o di unità di importazione, tipicamente > 70% della dimensione della stanza).

### Override manuali

Abilitare **Override manuali** nel pannello per accedere ai controlli per-oggetto. Premere **Sincronizza** per popolare la lista con gli oggetti della scena corrente. La sincronizzazione aggiunge le voci mancanti con classificazione automatica e rimuove quelle di oggetti non più presenti, preservando le voci già modificate manualmente.

Per ogni oggetto la lista espone quattro controlli:

| Controllo | Icona | Effetto |
|---|---|---|
| Toggle etichetta | Occhio | Include o esclude il nome dell'oggetto dai render etichettati |
| Ricerca padre | Catena | Assegna questo oggetto come figlio rigido di un altro |
| Toggle fisso | Lucchetto | Forza l'oggetto come immobile (ignora la classificazione automatica) |
| Toggle keep scale | Puntina | Esclude questo oggetto dall'operatore Scala a misura reale |

**Rileva automaticamente** riesegue la classificazione automatica fisso/mobile su tutte le voci, sovrascrivendo le scelte manuali sul toggle fisso.

**Suggerisci gruppi** analizza la geometria della scena e propone relazioni padre-figlio nei campi Padre della lista. La logica è puramente geometrica, senza categorie semantiche:

- Un oggetto è candidato figlio se la sua impronta XY si sovrappone (verificato via SAT su OBB) a quella di un candidato padre che ha un'impronta o un volume significativamente maggiore.
- La condizione verticale richiede che il figlio sia "sopra" il padre (con tolleranza di 8 cm sotto e 20 cm sopra il bordo superiore del padre) oppure che stia dentro il padre (con tolleranza di 5 cm su entrambi i bordi Z) oppure che condivida almeno il 30% dell'altezza con il padre se il padre è molto più grande in volume.
- Tra più candidati padri che soddisfano i criteri, viene scelto quello con lo score migliore (distanza verticale minima per la condizione "sopra", distanza orizzontale per la condizione "vicino").

I suggerimenti compaiono nel campo Padre della lista e possono essere accettati, modificati o ignorati.

### Step 1 — Randomize Layout

Premere **Randomize Layout**. L'add-on:

1. Al primo click, salva la posa corrente di ogni oggetto come "stato originale" in custom property Blender (`nl2_home_loc`, `nl2_home_rot`) che persistono nel file `.blend`. Salva anche uno snapshot `m_orig` per le metriche di spostamento.
2. Chiama `scene_io.extract_scene_state()` per costruire lo `SceneState` corrente.
3. Istanzia `SceneRandomizer` con il seed configurato nelle preferenze (0 = casuale, qualsiasi intero positivo = riproducibile).
4. Il randomizer ordina i root mobili per volume decrescente e per ognuno:
   a. Genera una rotazione Z casuale come multiplo di 90° rispetto alla rotazione originale.
   b. Genera una posizione XY casuale nell'area sicura (dentro i confini meno `wall_margin`, tenendo conto dell'AABB ruotato e dell'origin offset), centrata sulla posizione originale con raggio `jitter_ratio × dimensione_stanza`.
   c. Calcola il collision score della posizione candidata (padre + tutti i discendenti trasformati rigidamente).
   d. Mantiene il candidato con score più basso tra tutti i tentativi. Se score = 0.0 viene accettato immediatamente senza esaurire il budget.
   e. Dopo aver scelto la posizione, clamp del gruppo intero dentro i confini (corregge gli overflow dopo la rotazione).
   f. Applica la trasformazione rigida a tutti i discendenti (ricorsivamente), preservando la Z originale di ognuno.
5. Chiama `scene_io.apply_state()` per scrivere le nuove pose in Blender.
6. Salva uno snapshot `m_rand` per le metriche.

### Step 2a — Render con etichette

Premere **2a. Render con etichette (opzionale)**. La pipeline produce:

- Una **vista prospettica d'angolo** con camera posizionata automaticamente in alto e di lato per inquadrare l'intera scena (distanza calcolata dal raggio della scena e dall'angolo di campo della focale scelta, default 24 mm). In alternativa si può usare la camera esistente nella scena.
- Una **vista ortografica dall'alto** (pianta), con scala ortografica pari alla dimensione maggiore dell'AABB degli oggetti etichettati × 1.10.
- Opzionalmente, una **vista isometrica** (ortografica obliqua a 45°).

Le opzioni di rendering (vista dall'alto, vista isometrica, auto-inquadratura, lente prospettica e luminosità) sono proprietà dell'operatore. Si configurano nel pannello **Modifica l'ultima operazione** (`Adjust Last Operation`, in basso a sinistra nel viewport, richiamabile con `F9`) che appare subito dopo aver premuto il pulsante. La vista isometrica è disattivata di default e va abilitata da questa sezione. Nel flusso automatico tramite API, la vista isometrica non viene generata.

L'inquadratura è sempre calcolata sull'AABB degli oggetti etichettati, non dell'intera scena, in modo che mesh lontani e non etichettati non espandano inutilmente il campo visivo.

Il render è eseguito via OpenGL/Workbench con illuminazione Studio e fattore di intensità aumentato (1.4), in modo che la scena sia sempre visibile anche con materiali scuri. Se OpenGL non è disponibile, si usa il renderer attivo come fallback.

Il post-processing (via Pillow) applica correzione di luminosità (default ×1.5) e correzione gamma (default 1.6), poi aggiunge:

- **Etichette in gutter layout:** ogni nome di oggetto visibile nel render viene posizionato nel margine esterno attorno all'immagine (il bordo più vicino all'oggetto), connesso da una linea di richiamo con punto di ancoraggio sull'oggetto. Le etichette sullo stesso bordo sono distribuite ordinatamente senza sovrapporsi. Il canvas viene allargato per ospitare le etichette senza coprire la scena.
- **Bussola degli assi X/Y** (X in rosso, Y in verde), posizionata in alto a destra, in tutte le viste.
- **Barra di scala metrica**, posizionata in basso a sinistra, solo nelle viste ortografiche dove un pixel corrisponde sempre alla stessa distanza in metri. La lunghezza della barra è arrotondata al valore "tondo" più vicino della serie 1-2-5 (come nelle scale cartografiche).

I PNG sono salvati in `nl2_renders/` accanto al file `.blend`, o nella directory temporanea di sistema se il file non è stato ancora salvato.

### Istruzioni personalizzate per l'LLM

Il campo Istruzioni LLM nel pannello (proprietà di scena `nl2_custom_instructions`) permette di aggiungere linee guida testuali libere al prompt, ad esempio "Metti la sedia davanti alla scrivania" o "Lascia libero l'angolo vicino alla finestra". Se compilato, il testo viene accodato al prompt sotto una sezione **## Custom User Guidelines** da `build_prompt()`. Le istruzioni vengono applicate a entrambi i flussi — sia all'esportazione manuale del prompt sia alla chiamata API automatica — perché entrambi passano per lo stesso `build_prompt(state, custom_instructions=...)`. Il campo è opzionale: se vuoto, il prompt resta quello generico.

### Riordino automatico via API

Il pulsante **Riordina automaticamente (API)** esegue l'intero ciclo con un solo click, senza copia-incolla: l'add-on renderizza le viste etichettate, contatta il provider LLM configurato nelle preferenze, allega le immagini, riceve la risposta e la applica con la stessa pipeline di sanitizzazione del flusso manuale. È implementato dall'operatore `NL2SCENE3D_OT_reorganize_with_api`. 
Sequenza:

  1. Se non esiste ancora, salva lo stato originale (`capture_home_state()`).
  2. Chiama `scene_io.extract_scene_state()` e verifica che esistano oggetti mobili root.
  3. Costruisce il prompt con `reorganizer.build_prompt()`, includendo le eventuali Istruzioni LLM, e lo scrive nel Text datablock `NL2_AI_Prompt`.
  4. Se il toggle **Renderizza prima della chiamata** è attivo (default), produce render etichettati freschi e li allega; ogni immagine è etichettata in modo che il modello sappia quale vista sta guardando (top-down, prospettica d'angolo, isometrica). Se il render fallisce, la chiamata prosegue comunque con il solo JSON.
  5. Invia la richiesta tramite `llm_providers.call_llm()`. Nella UI la chiamata gira in un thread di background con loop modale, così Blender non si congela durante l'attesa; richiamato da script, l'operatore lavora invece in modo sincrono (bloccante).
  6. Salva la risposta grezza nel Text datablock di risposta e la applica con `sanitize_response()`.


I provider supportati sono **Google Gemini**, **Anthropic** e **OpenAI**; modello e parametri della chiamata si impostano nelle preferenze (vedi [Configurazione](#configurazione)). Quando il provider lo consente, la richiesta forza l'output in JSON, con `extract_json()` come rete di sicurezza a valle; gli errori temporanei (limiti di rate, sovraccarico, blip di rete) vengono ritentati automaticamente.


### Step 2b — Esporta prompt per LLM

Premere **2b. Esporta prompt (copia negli appunti)**. L'add-on:

1. Chiama `scene_io.extract_scene_state()` per lo stato corrente.
2. Chiama `reorganizer.build_request()` per costruire il payload JSON: la stanza (`room`), gli ostacoli fissi non strutturali (`fixed_objects`) e gli oggetti mobili root (`movable_objects`). Per ogni mobile root vengono inclusi il centro geometrico del gruppo (gcx, gcy), l'impronta XY dell'intero gruppo (padre + tutti i discendenti), la rotazione corrente e, opzionalmente, la lista dei figli come contesto read-only.
3. Genera il prompt completo unendo le istruzioni con il payload JSON.
4. **Copia il prompt generato direttamente negli appunti del tuo sistema operativo** (e per backup lo scrive anche nel Text datablock `NL2_AI_Prompt`).

Ti basterà andare sul tuo LLM di riferimento e premere **Ctrl + V** per incollare il prompt pronto. Allega i render delle viste prodotti al punto 2 e invia il messaggio.

### Step 2c — Applica la risposta dell'LLM

Il modello deve rispondere con un JSON nella forma:

```json
{
  "placements": [
    { "name": "furniture_bed", "x": -2.5, "y": 1.0, "rotation_deg": 0 },
    { "name": "furniture_desk", "x": 3.8, "y": -2.0, "rotation_deg": 90 }
  ]
}
```

Per applicare la risposta, puoi usare il metodo principale basato su appunti o uno dei metodi alternativi:

- **Metodo degli appunti (Consigliato - 1 Click):**
  1. Seleziona e copia il JSON generato dall'LLM negli appunti del sistema.
  2. In Blender, clicca su **`2c. Applica risposta (dagli appunti)`**. L'add-on leggerà direttamente la risposta dagli appunti e organizzerà la scena.

- **Metodo da Text Editor (Alternativa):**
  Incolla il JSON del modello nel Text datablock `NL2_AI_Response` in Blender, quindi clicca su **Da Text Editor**.

- **Metodo da file (Alternativa):**
  Salva la risposta come file `.json` o `.txt` e clicca su **Da file** per caricarla tramite il file browser di Blender.

In tutti i casi viene eseguita la stessa pipeline di sanitizzazione e verifica delle collisioni e confini.

### Reset allo stato originale

**Reset to Original** riporta ogni oggetto alla posa salvata al primo click su Randomize. Lo stato è non-distruttivo: persiste nelle custom property di Blender e sopravvive al salvataggio del file. Il reset è sempre disponibile finché esiste lo snapshot originale nella scena.

### Metriche di spostamento

Dopo ogni applicazione di risposta LLM, il Text datablock `NL2_Metrics` viene aggiornato con una tabella che confronta tre stati:

- **O (Originale):** la posa al momento del primo Randomize.
- **R (Randomizzato):** la posa dopo il Randomize.
- **C (Corrente):** la posa dopo l'applicazione della risposta LLM.

Le colonne mostrano la distanza di spostamento XY O→R (quanto è stato disordinato), la distanza R→C (quanto ha mosso l'LLM) e la variazione di rotazione attorno a Z R→C in gradi. Il report include anche il totale e la media degli spostamenti R→C.

---

## Classificazione automatica degli oggetti

La classificazione automatica assegna a ogni oggetto una categoria e uno stato fisso/mobile:

- **`technical` / fisso:** oggetti di tipo non-MESH in Blender (CAMERA, LIGHT, SPEAKER, ARMATURE, EMPTY, CURVE). Non partecipano mai alla collision detection.
- **`structural` / fisso:** oggetti MESH il cui nome contiene almeno una delle parole chiave strutturali (confronto su token interi per evitare falsi positivi: "doorknob" non corrisponde a "door", "window_blind" non corrisponde a "window"). Le parole chiave sono definite in `STRUCTURAL_PATTERNS` in `settings.py` e coprono italiano e inglese: `wall`, `floor`, `ceiling`, `room`, `door`, `window`, `muro`, `parete`, `pavimento`, `soffitto`, `porta`, `finestra`, `stanza`.
- **`object` / mobile:** tutto il resto.

La classificazione automatica viene sempre rispettata per `technical`; per `structural` e `object` può essere sovrascritta dall'utente tramite il toggle fisso/mobile nel pannello override. `resolve_classification()` prende in input l'override dell'utente e, se presente, usa il campo `"fixed"` per determinare `is_movable`, altrimenti usa la stima automatica.

---

## Calcolo dei confini della stanza

`compute_room_bounds()` calcola i `RoomBounds` a partire dalla lista degli oggetti della scena secondo questa strategia a cascata:

1. Se esistono oggetti con categoria `structural`, vengono usati come riferimento.
2. Se non esistono strutturali riconosciuti per nome, viene usato l'oggetto MESH con l'impronta XY maggiore (il più probabile candidato a essere la mesh-stanza).
3. Se un singolo oggetto strutturale domina il volume (oltre il 50% del totale e oltre 1 m³), viene usato il suo AABB direttamente come confini della stanza.
4. Altrimenti si calcola l'AABB unione di tutti gli oggetti source.

La quota del soffitto (`z_ceiling`) viene dedotta dagli oggetti il cui nome contiene "ceiling", "room", "roof", "soffitto" o "stanza"; se nessuno corrisponde, viene usato il massimo Z rilevato tra gli strutturali (con un minimo di 2.5 m se il valore sarebbe inferiore a 1 m).

---

## Sistema di grouping padre-figlio

I gruppi padre-figlio vengono definiti dall'utente tramite il pannello override (o proposti da **Suggerisci gruppi**) e applicati da `apply_manual_parents()`. Le relazioni cicliche banali (A→B e B→A) vengono silenziosamente ignorate; lo stesso vale per riferimenti a oggetti inesistenti e per l'auto-riferimento.

Una volta definiti i gruppi, il sistema li tratta come blocchi rigidi in tutte le operazioni:

- **Randomize:** il randomizer posiziona solo i root (oggetti senza padre); i figli seguono con trasformazione rigida XY. Per il collision score, l'intero gruppo viene valutato insieme: il punteggio del root candidato più quello di tutti i discendenti trasformati rigidamente.
- **Sanitizzazione:** il modello produce posizioni solo per i root. I figli vengono aggiornati in `sanitize_response()` nella Fase 3, dopo la risoluzione delle collisioni.
- **Clamping:** il clamp usa l'AABB combinato del gruppo (calcolato da `group_aabb_xy()` in `geometry.py`) per garantire che padre e tutti i figli restino dentro i confini.
- **Payload JSON:** `build_request()` include i figli solo come campo `contains` read-only nel payload del padre, per dare contesto al modello senza chiedergli di posizionarli.

---

## Algoritmo di randomizzazione

`SceneRandomizer.randomize()` lavora su una copia profonda dello `SceneState`; l'originale non viene mai modificato.

Algoritmo per ogni root mobile (in ordine di volume decrescente):

1. Valuta la posizione corrente come candidato iniziale (score iniziale).
2. Per `max_placement_attempts` iterazioni (default 200):
   a. Genera una rotazione Z casuale: rotazione originale + uno tra {0°, 90°, 180°, 270°} scelto uniformemente.
   b. Genera una posizione XY: calcola il range sicuro tenendo conto dell'AABB ruotato e dell'origin offset; restringe il range a `jitter_ratio × dimensione_stanza` attorno alla posizione originale; se il range ristretto è invalido, usa il range sicuro completo.
   c. Calcola `collision_score()` per il root candidato + tutti i discendenti trasformati rigidamente.
   d. Se score = 0.0, accetta immediatamente. Altrimenti aggiorna il best se score < best_score.
3. Clamp del gruppo: corregge gli overflow rispetto ai confini dopo la rotazione proposta.
4. Applica la posizione al root. Z intatta (ripristinata esplicitamente dopo ogni operazione).
5. Sposta i figli con `apply_rigid_transform()`, preservando la Z originale di ognuno.

Il `collision_score()` è una somma pesata: penalità fissa di 100 + proporzione di overflow per contenimento fuori dai confini; penalità di 50 per invasione della clearance di una porta; 25 per invasione della clearance di una finestra; `aabb_overlap_ratio` × 2 per sovrapposizione con muri; `aabb_overlap_ratio` × 1 per sovrapposizione con altri mobili.

---

## Pipeline di render

`render_labeled_views()` salva e ripristina tutte le impostazioni di render di Blender (risoluzione, filepath, formato, camera) ed elimina le camera temporanee create al termine, anche in caso di eccezione.

Le camera temporanee vengono create e collegate alla collezione della scena, poi rimosse al termine insieme ai loro datablock camera. La risoluzione è sempre quadrata (`render_edge_px × render_edge_px`, default 1024×1024).

Il calcolo delle direzioni schermo degli assi X/Y per la bussola usa `world_to_camera_view()` di Blender: proietta il centro della scena e un punto spostato di 0.5 m lungo X e uno lungo Y, calcola le coordinate pixel corrispondenti e normalizza la differenza.

Il gutter layout calcola prima il bordo più vicino per ogni etichetta (usando la distanza dell'ancora dal bordo dell'immagine), poi allarga il canvas aggiungendo margini proporzionali alla larghezza massima delle etichette per quel bordo, e infine distribuisce le etichette di ogni bordo ordinatamente lungo l'asse del bordo stesso, centrandole nell'intervallo disponibile e separandole di un gap minimo.

---

## Costruzione del prompt e del payload JSON

`build_request()` include nel payload:

- `room`: i quattro confini della stanza (x_min, x_max, y_min, y_max) in metri, arrotondati a 3 decimali.
- `fixed_objects`: gli oggetti fissi non strutturali (fissi per override utente o perché non-MESH), ognuno con nome, centro geometrico e impronta XY. Gli strutturali (muri, pavimento, ecc.) sono esclusi perché già impliciti nei confini della stanza.
- `movable_objects`: i root mobili, ognuno con nome, centro geometrico del gruppo (gcx, gcy), impronta XY dell'intero gruppo (padre + tutti i discendenti), rotazione corrente in gradi interi (% 360) e lista `contains` dei figli con le loro impronte.

L'impronta di ogni gruppo è calcolata da `group_aabb_xy()` nella posa corrente, quindi tiene conto della rotazione e di tutti i figli nella loro posizione relativa attuale. Il modello riceve l'impronta del gruppo come un unico blocco rigido di dimensioni `w × d`; non deve conoscere la struttura interna del gruppo.

`build_prompt(state, custom_instructions="")` assembla il prompt finale: il template generico `PROMPT_TEMPLATE`, seguito dal payload JSON prodotto da `build_request()` in un blocco  ```json ```, e — solo se l'utente ha compilato il campo Istruzioni LLM — da una sezione `## Custom User Guidelines` con il testo libero fornito. La stessa funzione è usata identica dal flusso manuale e da quello automatico via API.

---

## Sanitizzazione della risposta LLM

`sanitize_response()` esegue tre fasi in sequenza:

**Fase 1 — Parsing e clamping.** Per ogni root mobile, applica la posizione proposta (o mantiene quella originale se il nome non compare nelle proposte). La rotazione viene convertita da gradi a radianti e poi snappata al multiplo di 90° più vicino (`snap_rotation_90()`). La rotazione viene impostata prima del clamping perché l'AABB ruotato deve essere corretto. Il clamping di gruppo usa `_clamp_parent_group_location()` per correggere i soli overflow: calcola l'AABB combinato del gruppo alla posizione proposta, misura separatamente gli overflow su ogni lato e aggiunge la correzione minima necessaria alla posizione del padre. La Z originale viene esplicitamente ripristinata dopo ogni operazione.

**Fase 2 — Risoluzione delle collisioni (MTV).** Itera fino a convergenza (o fino a `max_iter=80`):
- Mobili vs ostacoli fissi: il mobile viene spostato interamente del vettore di penetrazione.
- Mobili vs mobili: la spinta viene divisa a metà tra i due oggetti.
Dopo ogni spostamento MTV, l'oggetto viene riclampato dentro i confini della stanza. La convergenza è raggiunta quando nessuna coppia presenta più penetrazione.

**Fase 3 — Aggiornamento dei figli.** Per ogni root, ripristina la posa originale del figlio (dallo snapshot pre-sanitizzazione), poi applica la trasformazione rigida relativa al nuovo padre, preservando la Z originale del figlio.

Il metodo `extract_json()` gestisce robustamente le risposte "sporche" del modello: rimuove i recinti Markdown (` ```json ... ``` `), individua il primo `{` nel testo, poi scorre carattere per carattere tenendo traccia della profondità delle graffe e dello stato "dentro una stringa" (con gestione degli escape) per trovare l'esatto punto di chiusura del primo oggetto JSON valido.

---

## Garanzie geometriche

Le seguenti invarianti sono garantite dal codice indipendentemente dall'output del modello:

- **La coordinata Z non viene mai modificata.** Nessun modulo del core scrive sulla componente Z di alcuna location, né per i root né per i figli.
- **Contenimento dentro la stanza.** Ogni AABB (tenendo conto della rotazione dell'oggetto) deve essere completamente contenuto in `RoomBounds` con almeno `wall_margin` di clearance su ogni lato (default 20 cm).
- **Clearance davanti alle aperture.** Una zona di rispetto di 90 cm viene mantenuta davanti a ogni porta; 50 cm davanti a ogni finestra. Gli oggetti che invaderebbero queste zone ricevono una penalità pesante durante la randomizzazione (50 punti per le porte, 25 per le finestre) e vengono spostati dal solver MTV dopo la sanitizzazione.
- **Assenza di sovrapposizioni tra mobili.** Il SAT viene applicato su OBB (non AABB) per gestire correttamente gli oggetti ruotati. Il margine di espansione di 5 cm per lato garantisce un gioco visivo minimo tra oggetti adiacenti.
- **Integrità dei gruppi rigidi.** Quando un padre viene spostato, tutti i discendenti (ricorsivamente) vengono riposizionati tramite trasformazione rigida XY che preserva offset e rotazioni relativi. L'AABB combinato del gruppo è usato per tutti i check di contenimento.
- **Snapping della rotazione.** Qualsiasi valore di `rotation_deg` proposto dal modello viene snappato al multiplo di 90° più vicino prima dell'applicazione.

---

## Configurazione

Tutte le costanti operative sono definite in `nl2scene3d_addon/core/settings.py` come dataclass frozen. Modificare il file e ricaricare l'add-on per aggiornare i valori; non è necessario riavviare Blender.

| Costante | Default | Descrizione |
|---|---|---|
| `wall_margin` | 0.20 m | Distanza minima tra qualsiasi oggetto mobile e un muro |
| `collision_margin` | 0.05 m | Espansione aggiuntiva applicata a ogni OBB durante i check mobile-mobile (ogni coppia ha almeno 10 cm di gioco visivo) |
| `jitter_ratio` | 0.80 | Ampiezza del disordine come frazione della dimensione della stanza; 1.0 = oggetti posizionati ovunque nella stanza, 0.0 = nessuno spostamento |
| `max_placement_attempts` | 200 | Budget di tentativi per ogni oggetto prima di usare il fallback (posizione con score più basso trovata) |
| `max_movable_objects` | 50 | Oggetti oltre questo limite vengono forzati a fisso; protegge da scene con centinaia di oggetti decorativi |
| `render_edge_px` | 1024 | Lato in pixel dei render quadrati prodotti e allegati al prompt |

Il seed del randomizer è esposto in **Modifica > Preferenze > Add-on > NL2Scene3D**. Seed 0 (default) produce un risultato diverso a ogni click; qualsiasi intero positivo produce un layout riproducibile, utile per confrontare le risposte di modelli diversi sulla stessa scena disordinata.

**Preferenze per la chiamata API automatica**

Le impostazioni del flusso automatico vivono nelle preferenze dell'add-on (**Modifica > Preferenze > Add-on > NL2Scene3D**), nel riquadro Chiamata API automatica. Sono usate solo dall'operatore **Riordina automaticamente (API)**; il flusso manuale le ignora.

| Preferenza | Default | Descrizione |
|---|---|---|
| `llm_provider` | Google Gemini | Provider da contattare: Gemini, Anthropic o OpenAI |
| `gemini_api_key` / `anthropic_api_key` / `openai_api_key` | | Chiave API del provider, salvata come PASSWORD. Viene mostrato solo il campo del provider selezionato. Se vuota, si usa la variabile d'ambiente corrispondente |
| `llm_model` | | Identificatore del modello. Vuoto = default del provider (`gemini-3.5-flash`, `claude-haiku-4-5`, `gpt-5.1-mini`) |
| `llm_temperature` | 0.7 | Creatività del modello (0 = deterministico; range 0–2) |
| `llm_timeout` | 120s | Tempo massimo di attesa della risposta (range 10–600 s) |
| `llm_max_retries` | 4 | Tentativi automatici extra su errori transitori 429/5xx/rete, con backoff esponenziale (range 0–10) |
| `llm_auto_render` | Attivo | Se attivo, genera e allega render etichettati freschi prima della chiamata |

Come fallback alle chiavi inserite nelle preferenze, l'add-on legge le variabili d'ambiente `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` e `OPENAI_API_KEY`. La chiave non viene mai trasmessa se non al provider selezionato.

---

## Testing offline

La directory `offline_testing/` contiene materiale per testare il sistema senza aprire Blender.

`PROMPT+JSONSCENE.txt` è un esempio completo di prompt + payload JSON relativo a una camera da letto con 12 oggetti (letto, scrivania, sedia, comodino, scaffale, decorazioni). Il file può essere inviato direttamente a qualsiasi LLM per valutare la qualità del layout proposto, confrontare modelli diversi o sviluppare varianti del prompt.

`PYTHON - SCRIPT.txt` è uno script minimale da eseguire nella console Python di Blender. Contiene un JSON di layout hard-coded e applica le posizioni direttamente agli oggetti della scena tramite `bpy`, bypassando completamente l'add-on. È utile per test rapidi di singole risposte o per valutazioni batch in cui l'overhead dell'interfaccia non è desiderato.

Poiché `nl2scene3d_addon/core/` non ha dipendenze da `bpy`, il core è esercitabile da riga di comando:

```python
import sys
sys.path.insert(0, "/percorso/a/nl2scene3d")  # aggiungere la root del progetto al path

from nl2scene3d_addon.core.models import Transform, SceneObject, RoomBounds, SceneState
from nl2scene3d_addon.core.reorganizer import build_request, build_prompt, extract_json, sanitize_response
from nl2scene3d_addon.core.randomizer import SceneRandomizer

# Costruire uno SceneState di test, invocare build_prompt(), simulare una risposta,
# chiamare sanitize_response() e verificare le pose risultanti senza Blender.
```

Anche `llm_providers.py` è importabile e testabile da riga di comando senza Blender (non dipende da `bpy`): si può invocare `call_llm()` con un prompt e immagini reali per verificare l'integrazione con un provider, indipendentemente dall'add-on.

---

## Struttura del progetto

```
.
├── nl2scene3d_addon/          Add-on Blender (installabile come zip)
│   ├── __init__.py
│   ├── operators.py
│   ├── ui.py
│   ├── llm_providers.py
│   └── core/
│       ├── __init__.py
│       ├── classify.py
│       ├── geometry.py
│       ├── models.py
│       ├── randomizer.py
│       ├── render.py
│       ├── reorganizer.py
│       ├── scene_io.py
│       └── settings.py
├── offline_testing/
│   ├── PROMPT+JSONSCENE.txt
│   └── PYTHON - SCRIPT.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedere [LICENSE](LICENSE) per il testo completo.