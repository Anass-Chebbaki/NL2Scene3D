# nl2scene3d/core/render.py
"""
Render automatico delle viste della scena con sovrapposizione di etichette,
bussola degli assi e barra di scala metrica.

Comportamento generale:
    - Il render viene eseguito tramite OpenGL/Workbench: la scena e' sempre
      illuminata (luminosita' + correzione gamma), mai completamente nera.
    - Vengono prodotte tre tipologie di vista:
        * Prospettica d'angolo auto-inquadrata (mostra l'intera stanza).
        * Pianta ortografica dall'alto (top-down).
        * Vista isometrica (opzionale).
    - Le etichette degli oggetti sono posizionate con un algoritmo anti-
      sovrapposizione (declutter) e collegate all'oggetto con linee di richiamo.
    - In ogni vista viene sovrapposta una bussola degli assi X/Y.
    - La barra di scala metrica appare SOLO nelle viste ortografiche (top-down
      e isometrica), dove un pixel corrisponde sempre alla stessa distanza in
      metri. Nella vista prospettica la barra sarebbe imprecisa, quindi e'
      omessa: compare solo la bussola.

Note di progetto:
    - Il modulo e' generico: nessun nome o tipo di stanza e' codificato.
    - La proiezione 3D->2D, il declutter e il calcolo del "nice number" per la
      barra sono funzioni pure (non dipendono da bpy).
    - La coordinata Z non viene mai modificata.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from typing import Optional

from .settings import CONST, Constants

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Funzioni pure di utilita'
# ---------------------------------------------------------------------------

def _ndc_to_pixels(ndc_x: float, ndc_y: float, width: int, height: int) -> tuple[int, int]:
    """
    Converte coordinate NDC (Normalized Device Coordinates, range [0,1]) in
    coordinate pixel dello schermo.

    L'asse Y e' invertito: NDC 0.0 corrisponde al bordo inferiore dell'immagine,
    mentre in pixel 0 e' il bordo superiore.

    Args:
        ndc_x: Coordinata X normalizzata.
        ndc_y: Coordinata Y normalizzata.
        width: Larghezza dell'immagine in pixel.
        height: Altezza dell'immagine in pixel.

    Returns:
        Coppia (px, py) di coordinate intere in pixel.
    """
    return int(round(ndc_x * width)), int(round((1.0 - ndc_y) * height))


def _is_in_frame(cam_x: float, cam_y: float, cam_z: float, margin: float = 0.02) -> bool:
    """
    Verifica se un punto proiettato e' visibile all'interno del fotogramma
    della camera, con un margine di tolleranza opzionale oltre i bordi.

    Args:
        cam_x: Coordinata X nello spazio camera (NDC).
        cam_y: Coordinata Y nello spazio camera (NDC).
        cam_z: Profondita' nello spazio camera; deve essere positiva.
        margin: Margine di tolleranza oltre i bordi [0,1] del fotogramma.

    Returns:
        True se il punto e' nel fotogramma (o entro il margine), False altrimenti.
    """
    return cam_z > 0.0 and -margin <= cam_x <= 1.0 + margin and -margin <= cam_y <= 1.0 + margin


def _safe(name: str) -> str:
    """
    Restituisce una versione del nome adatta all'uso come componente di un
    percorso file: mantiene caratteri alfanumerici, trattini e underscore;
    sostituisce tutto il resto con underscore.

    Args:
        name: Stringa originale.

    Returns:
        Stringa sanitizzata.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _nice_length(raw_m: float) -> float:
    """
    Arrotonda una lunghezza in metri al valore "tondo" piu' vicino della serie
    1, 2, 5 (moltiplicati per potenze di 10). Usata per scegliere la lunghezza
    della barra di scala in modo che il numero visualizzato sia leggibile.

    Esempi:
        0.37 -> 0.5
        1.4  -> 1.0
        2.8  -> 2.0
        6.1  -> 5.0

    Args:
        raw_m: Lunghezza grezza in metri.

    Returns:
        Lunghezza arrotondata al valore "tondo" piu' appropriato.
    """
    if raw_m <= 0:
        return 1.0
    k = math.floor(math.log10(raw_m))
    base = raw_m / (10 ** k)
    nice = 1 if base < 1.5 else (2 if base < 3.5 else 5)
    return nice * (10 ** k)


def _declutter(boxes: list, width: int, height: int, pad: int = 5, iterations: int = 150) -> list:
    """
    Separa i riquadri delle etichette finche' non si sovrappongono piu',
    usando il Minimum Translation Vector (MTV) in spazio pixel.

    Ad ogni iterazione, per ogni coppia di riquadri sovrapposti viene
    calcolata la direzione e la quantita' minima di spostamento necessaria a
    eliminarli (MTV); la spinta viene divisa a meta' tra i due. I riquadri
    vengono poi mantenuti dentro i bordi dell'immagine.

    Args:
        boxes:      Lista di dict con chiavi 'cx', 'cy', 'w', 'h'.
        width:      Larghezza dell'immagine in pixel.
        height:     Altezza dell'immagine in pixel.
        pad:        Spazio minimo aggiuntivo (in pixel) tra due riquadri.
        iterations: Numero massimo di iterazioni prima di interrompere.

    Returns:
        La lista di box aggiornata con le nuove posizioni 'cx', 'cy'.
    """
    for _ in range(iterations):
        moved = False

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                dx = b["cx"] - a["cx"]
                dy = b["cy"] - a["cy"]

                # Evita la divisione per zero se i centri coincidono.
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    dy = 1.0

                # Overlap sulle due assi.
                ox = (a["w"] + b["w"]) / 2.0 + pad - abs(dx)
                oy = (a["h"] + b["h"]) / 2.0 + pad - abs(dy)

                if ox > 0 and oy > 0:
                    # Sposta lungo l'asse con il minor overlap (MTV).
                    if ox < oy:
                        s = ox / 2.0 * (1.0 if dx >= 0 else -1.0)
                        a["cx"] -= s
                        b["cx"] += s
                    else:
                        s = oy / 2.0 * (1.0 if dy >= 0 else -1.0)
                        a["cy"] -= s
                        b["cy"] += s
                    moved = True

        # Mantieni ogni box entro i bordi dell'immagine.
        for bx in boxes:
            hw, hh = bx["w"] / 2.0, bx["h"] / 2.0
            bx["cx"] = min(max(bx["cx"], hw), width - hw)
            bx["cy"] = min(max(bx["cy"], hh), height - hh)

        if not moved:
            break

    return boxes


# ---------------------------------------------------------------------------
# Layout delle etichette nella cornice (gutter layout)
# ---------------------------------------------------------------------------

def _gutter_layout(boxes: list, W: int, H: int, gap: int = 10, edge: int = 8):
    """
    Posiziona le etichette nella cornice attorno all'immagine della scena,
    assegnando ciascuna al bordo (L/R/T/B) geometricamente piu' vicino
    all'ancoraggio dell'oggetto corrispondente. Le etichette vengono poi
    distribuite ordinatamente lungo il bordo senza sovrapporsi.

    Funzione pura: non accede a bpy ne' alla rete.

    Scrittura sui box:
        - 'border': il lato assegnato ('L', 'R', 'T' o 'B').
        - 'cx', 'cy': centro del box nelle coordinate del canvas allargato.

    Args:
        boxes: Lista di dict con chiavi 'ax', 'ay' (ancora sull'immagine
               originale in pixel), 'w', 'h' (dimensioni del box etichetta).
        W:     Larghezza dell'immagine originale in pixel.
        H:     Altezza dell'immagine originale in pixel.
        gap:   Spazio minimo tra etichette consecutive sullo stesso bordo.
        edge:  Distanza minima dal bordo del canvas per la prima/ultima etichetta.

    Returns:
        Tupla (ml, mr, mt, mb, Wp, Hp) con i margini sinistro/destro/
        superiore/inferiore e le dimensioni totali del canvas allargato.
    """
    # Assegna ciascuna etichetta al bordo piu' vicino al suo oggetto.
    for b in boxes:
        dl, dr, dt, db = b["ax"], W - b["ax"], b["ay"], H - b["ay"]
        m = min(dl, dr, dt, db)
        b["border"] = "L" if m == dl else "R" if m == dr else "T" if m == dt else "B"

    L = [b for b in boxes if b["border"] == "L"]
    R = [b for b in boxes if b["border"] == "R"]
    T = [b for b in boxes if b["border"] == "T"]
    B = [b for b in boxes if b["border"] == "B"]

    # Calcola i margini necessari per ogni bordo.
    ml = (max(b["w"] for b in L) + 2 * gap) if L else gap
    mr = (max(b["w"] for b in R) + 2 * gap) if R else gap
    mt = (max(b["h"] for b in T) + 2 * gap) if T else gap
    mb = (max(b["h"] for b in B) + 2 * gap) if B else gap

    Wp = W + ml + mr
    Hp = H + mt + mb

    def pack(items: list, key, dim: str, lo: float, hi: float) -> None:
        """
        Distribuisce gli oggetti di 'items' lungo un asse, centrandoli
        nell'intervallo [lo, hi] e separandoli di 'gap' pixel.
        Scrive la coordinata risultante in '_pos'.
        """
        items = sorted(items, key=key)
        total = sum(b[dim] for b in items) + gap * max(0, len(items) - 1)
        cur = lo + max(0.0, (hi - lo - total)) / 2.0
        for b in items:
            b["_pos"] = cur + b[dim] / 2.0
            cur += b[dim] + gap

    # Posiziona le etichette di ciascun bordo nel canvas allargato.
    pack(L, lambda b: b["ay"], "h", edge, Hp - edge)
    for b in L:
        b["cx"] = ml - gap - b["w"] / 2.0
        b["cy"] = b["_pos"]

    pack(R, lambda b: b["ay"], "h", edge, Hp - edge)
    for b in R:
        b["cx"] = Wp - mr + gap + b["w"] / 2.0
        b["cy"] = b["_pos"]

    pack(T, lambda b: b["ax"], "w", edge, Wp - edge)
    for b in T:
        b["cy"] = mt - gap - b["h"] / 2.0
        b["cx"] = b["_pos"]

    pack(B, lambda b: b["ax"], "w", edge, Wp - edge)
    for b in B:
        b["cy"] = Hp - mb + gap + b["h"] / 2.0
        b["cx"] = b["_pos"]

    return ml, mr, mt, mb, Wp, Hp


# ---------------------------------------------------------------------------
# Disegno delle etichette sull'immagine (Pillow)
# ---------------------------------------------------------------------------

def _draw_labels(
    image_path: str,
    points: list,
    font_size: int = 18,
    brighten: float = 1.5,
    gamma: float = 1.6,
) -> bool:
    """
    Applica correzione di luminosita' e gamma all'immagine, quindi aggiunge
    le etichette degli oggetti nella cornice attorno alla scena. La scena
    rimane interamente visibile: nessuna etichetta e' sovrapposta all'immagine.
    Ogni etichetta e' collegata al suo oggetto da una linea di richiamo.

    Se Pillow non e' disponibile, le etichette vengono scritte in un file
    .labels.json accanto all'immagine, e la funzione restituisce False.

    Args:
        image_path: Percorso del file PNG da modificare (sovrascritta in-place).
        points:     Lista di tuple (nome, x_pixel, y_pixel) per ogni oggetto.
        font_size:  Dimensione del font in punti.
        brighten:   Fattore di luminosita' (1.0 = invariato).
        gamma:      Esponente di correzione gamma (1.0 = invariato).

    Returns:
        True se le etichette sono state disegnate, False in caso di errore.
    """
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFont  # type: ignore
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # Tollera PNG non completamente scritti.
    except ImportError:
        if points:
            sidecar = os.path.splitext(image_path)[0] + ".labels.json"
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump([{"name": n, "x": x, "y": y} for n, x, y in points], f, indent=2)
            logger.warning("Pillow non disponibile: etichette non disegnate, scritto %s", sidecar)
        return False

    base = Image.open(image_path).convert("RGB")

    # Correzione di luminosita'.
    if brighten and abs(brighten - 1.0) > 1e-3:
        base = ImageEnhance.Brightness(base).enhance(brighten)

    # Correzione gamma tramite LUT a 256 valori.
    if gamma and abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / gamma
        lut = [min(255, int(round(255.0 * ((i / 255.0) ** inv)))) for i in range(256)]
        base = base.point(lut * 3)

    if not points:
        base.save(image_path)
        return True

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    W, H = base.size
    measure = ImageDraw.Draw(base)

    # Calcola le dimensioni di ogni etichetta per il layout.
    boxes = []
    for name, x, y in points:
        tb = measure.textbbox((0, 0), name, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        boxes.append({
            "name": name,
            "ax": float(x), "ay": float(y),
            "w": tw + 10, "h": th + 8,
            "tw": tw, "th": th,
            "ox": tb[0], "oy": tb[1],
        })

    ml, mr, mt, mb, Wp, Hp = _gutter_layout(boxes, W, H)

    # Canvas allargato con sfondo grigio scuro; la scena viene incollata al centro.
    canvas = Image.new("RGB", (int(Wp), int(Hp)), (40, 40, 40))
    canvas.paste(base, (int(ml), int(mt)))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for b in boxes:
        # Posizione dell'ancora nel canvas allargato.
        ax, ay = b["ax"] + ml, b["ay"] + mt
        cx, cy = b["cx"], b["cy"]

        # Punto di partenza della linea di richiamo: bordo del box rivolto verso la scena.
        if b["border"] == "L":
            sx, sy = cx + b["w"] / 2.0, cy
        elif b["border"] == "R":
            sx, sy = cx - b["w"] / 2.0, cy
        elif b["border"] == "T":
            sx, sy = cx, cy + b["h"] / 2.0
        else:
            sx, sy = cx, cy - b["h"] / 2.0

        # Linea di richiamo e punto di ancoraggio.
        draw.line((sx, sy, ax, ay), fill=(255, 255, 255, 230), width=1)
        draw.ellipse((ax - 2, ay - 2, ax + 2, ay + 2), fill=(255, 255, 255, 255))

        # Riquadro dell'etichetta: sfondo nero opaco, testo bianco.
        x0, y0 = cx - b["w"] / 2.0, cy - b["h"] / 2.0
        draw.rectangle((x0, y0, x0 + b["w"], y0 + b["h"]), fill=(0, 0, 0, 255))
        tx = cx - b["tw"] / 2.0 - b["ox"]
        ty = cy - b["th"] / 2.0 - b["oy"]
        draw.text((tx, ty), b["name"], font=font, fill=(255, 255, 255, 255))

    Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB").save(image_path)
    return True


# ---------------------------------------------------------------------------
# Disegno dell'overlay (bussola assi + barra di scala)
# ---------------------------------------------------------------------------

def _draw_overlay(image_path: str, axes_dirs, meters_per_pixel: float = None) -> None:
    """
    Sovrappone all'immagine la bussola degli assi (X in rosso, Y in verde) e,
    per le sole viste ortografiche, la barra di scala metrica.

    La bussola e' posizionata in alto a destra; la barra di scala e' posizionata
    in basso a sinistra. Nella vista prospettica la barra non viene disegnata
    perche' la corrispondenza pixel/metro non e' costante.

    Args:
        image_path:       Percorso del file PNG da modificare (in-place).
        axes_dirs:        Coppia di vettori schermo ((xdx,xdy), (ydx,ydy))
                          che indicano la direzione degli assi +X e +Y.
        meters_per_pixel: Scala della vista ortografica (metri per pixel).
                          Se None o <= 0, la barra di scala non viene disegnata.
    """
    try:
        from PIL import Image, ImageDraw, ImageFile, ImageFont  # type: ignore
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    except ImportError:
        return

    img = Image.open(image_path).convert("RGBA")
    W, H = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        font_axis = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        font_axis = font

    # Bussola degli assi: posizionata in alto a destra.
    if axes_dirs:
        (xdx, xdy), (ydx, ydy) = axes_dirs
        ax, ay, L = W - 92, 92, 56

        draw.line((ax, ay, ax + xdx * L, ay + xdy * L), fill=(235, 70, 70, 255), width=5)
        draw.text(
            (ax + xdx * (L + 14) - 7, ay + xdy * (L + 14) - 12),
            "X", font=font_axis, fill=(235, 70, 70, 255),
        )
        draw.line((ax, ay, ax + ydx * L, ay + ydy * L), fill=(70, 200, 70, 255), width=5)
        draw.text(
            (ax + ydx * (L + 14) - 7, ay + ydy * (L + 14) - 12),
            "Y", font=font_axis, fill=(70, 200, 70, 255),
        )
        draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill=(255, 255, 255, 255))

    # Barra di scala: solo per le viste ortografiche (scala costante).
    if meters_per_pixel and meters_per_pixel > 0:
        bar_m = _nice_length(140 * meters_per_pixel)
        bar_px = int(round(bar_m / meters_per_pixel))
        x0, y0 = 30, H - 36

        # Sfondo semi-trasparente per leggibilita'.
        draw.rectangle((x0 - 7, y0 - 22, x0 + bar_px + 7, y0 + 10), fill=(0, 0, 0, 150))

        # Linea orizzontale con terminatori verticali.
        draw.line((x0, y0, x0 + bar_px, y0), fill=(255, 255, 255, 255), width=3)
        draw.line((x0, y0 - 6, x0, y0 + 6), fill=(255, 255, 255, 255), width=3)
        draw.line((x0 + bar_px, y0 - 6, x0 + bar_px, y0 + 6), fill=(255, 255, 255, 255), width=3)

        draw.text((x0, y0 - 20), f"{bar_m:g} m", font=font, fill=(255, 255, 255, 255))

    Image.alpha_composite(img, overlay).convert("RGB").save(image_path)


# ---------------------------------------------------------------------------
# Funzioni che richiedono il contesto Blender (bpy-facing)
# ---------------------------------------------------------------------------

def _ensure_output_dir(subdir: str = "nl2_renders") -> str:
    """
    Crea (se non esiste) e restituisce la directory di output per i render.
    Se il file .blend e' salvato, la directory e' relativa ad esso;
    altrimenti viene usata la directory temporanea di sistema.

    Args:
        subdir: Nome della sottodirectory di output.

    Returns:
        Percorso assoluto della directory di output.
    """
    import bpy  # noqa: PLC0415
    base = (
        bpy.path.abspath("//" + subdir + "/")
        if bpy.data.filepath
        else os.path.join(tempfile.gettempdir(), subdir)
    )
    os.makedirs(base, exist_ok=True)
    return base


def _objects_aabb(objs: list):
    """
    Calcola l'AABB (Axis-Aligned Bounding Box) in spazio mondo dell'unione
    di tutti i MESH nella lista fornita.

    Args:
        objs: Lista di oggetti Blender.

    Returns:
        Tupla (mins, maxs, center) come mathutils.Vector, oppure
        (zero, zero, zero) se nessun MESH e' presente nella lista.
    """
    import mathutils  # noqa: PLC0415

    big = 1.0e9
    mins = mathutils.Vector((big, big, big))
    maxs = mathutils.Vector((-big, -big, -big))
    found = False

    for o in objs:
        if getattr(o, "type", None) != "MESH":
            continue
        found = True
        for c in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(c)
            mins = mathutils.Vector((min(mins.x, w.x), min(mins.y, w.y), min(mins.z, w.z)))
            maxs = mathutils.Vector((max(maxs.x, w.x), max(maxs.y, w.y), max(maxs.z, w.z)))

    if not found:
        z = mathutils.Vector((0.0, 0.0, 0.0))
        return z, z, z

    return mins, maxs, (mins + maxs) / 2.0


def _scene_aabb(scene, objs: list = None):
    """
    Restituisce l'AABB di inquadratura della scena.

    Se 'objs' e' fornito, l'AABB viene calcolato solo su quegli oggetti
    (tipicamente quelli etichettati), in modo che mesh lontani e non
    etichettati non gonfino inutilmente la vista. Se 'objs' e' None,
    vengono considerati tutti i MESH della scena.

    Args:
        scene: La scena Blender corrente.
        objs:  Lista opzionale di oggetti su cui calcolare l'AABB.

    Returns:
        Tupla (mins, maxs, center) come mathutils.Vector.
    """
    if objs:
        return _objects_aabb(objs)
    return _objects_aabb(list(scene.objects))


def _axes_screen_dirs(scene, cam, width: int, height: int, center):
    """
    Calcola le direzioni schermo (vettori unitari 2D) degli assi mondo +X e +Y
    visti dalla camera specificata. Usato per disegnare la bussola.

    Args:
        scene:  La scena Blender.
        cam:    L'oggetto camera attivo.
        width:  Larghezza del render in pixel.
        height: Altezza del render in pixel.
        center: Punto centrale della scena in spazio mondo (mathutils.Vector o sequenza).

    Returns:
        Coppia di tuple ((xdx, xdy), (ydx, ydy)) con le direzioni normalizzate
        degli assi +X e +Y in coordinate schermo.
    """
    import mathutils  # noqa: PLC0415
    from bpy_extras.object_utils import world_to_camera_view  # noqa: PLC0415

    C = mathutils.Vector(center)
    k = 0.5  # Offset di campionamento per la direzione degli assi.

    def to_px(v):
        co = world_to_camera_view(scene, cam, v)
        return _ndc_to_pixels(co.x, co.y, width, height)

    def unit(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = (dx * dx + dy * dy) ** 0.5
        return (1.0, 0.0) if n < 1e-6 else (dx / n, dy / n)

    p0 = to_px(C)
    return (
        unit(p0, to_px(C + mathutils.Vector((k, 0, 0)))),
        unit(p0, to_px(C + mathutils.Vector((0, k, 0)))),
    )


def _label_points(scene, cam, mesh_objs: list, width: int, height: int) -> list:
    """
    Calcola le coordinate pixel dei centri geometrici degli oggetti MESH,
    proiettati nella vista della camera corrente. Gli oggetti fuori campo
    vengono esclusi.

    Args:
        scene:     La scena Blender.
        cam:       L'oggetto camera attivo.
        mesh_objs: Lista di oggetti MESH da etichettare.
        width:     Larghezza del render in pixel.
        height:    Altezza del render in pixel.

    Returns:
        Lista di tuple (nome, x_pixel, y_pixel) per gli oggetti visibili.
    """
    import mathutils  # noqa: PLC0415
    from bpy_extras.object_utils import world_to_camera_view  # noqa: PLC0415

    pts = []
    for ob in mesh_objs:
        local_center = sum(
            (mathutils.Vector(c) for c in ob.bound_box), mathutils.Vector()
        ) / 8.0
        world_center = ob.matrix_world @ local_center
        co = world_to_camera_view(scene, cam, world_center)
        if not _is_in_frame(co.x, co.y, co.z):
            continue
        px, py = _ndc_to_pixels(co.x, co.y, width, height)
        pts.append((ob.name, px, py))

    return pts


def _render_to(path: str) -> None:
    """
    Esegue il render della vista corrente e salva il risultato nel percorso
    indicato. Tenta prima il render OpenGL (Workbench) con illuminazione
    studio; se non disponibile, ricade sul render standard del motore attivo.

    Args:
        path: Percorso di output del file PNG.
    """
    import bpy  # noqa: PLC0415

    scene = bpy.context.scene
    scene.render.filepath = path

    try:
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
        if hasattr(shading, "studiolight_intensity"):
            shading.studiolight_intensity = 1.4
    except Exception:
        pass

    try:
        bpy.ops.render.opengl(write_still=True, view_context=False)
    except RuntimeError:
        bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Creazione delle camera temporanee
# ---------------------------------------------------------------------------

def _make_top_down_camera(scene, aabb=None):
    """
    Crea una camera ortografica dall'alto (pianta), inquadrata sull'AABB
    fornito o sull'intera scena se aabb e' None.

    Args:
        scene: La scena Blender.
        aabb:  Tupla opzionale (mins, maxs, center) da _objects_aabb/_scene_aabb.

    Returns:
        Il nuovo oggetto camera, gia' collegato alla collezione della scena.
    """
    import bpy  # noqa: PLC0415

    mins, maxs, center = aabb if aabb is not None else _scene_aabb(scene)

    cam_data = bpy.data.cameras.new("NL2_TopDown")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(maxs.x - mins.x, maxs.y - mins.y, 1.0) * 1.10

    cam = bpy.data.objects.new("NL2_TopDown", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (center.x, center.y, maxs.z + 5.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)

    return cam


def _make_corner_camera(scene, lens_mm: float = 24.0, aabb=None):
    """
    Crea una camera prospettica d'angolo, posizionata in alto e di lato in
    modo da inquadrare l'intera stanza. La distanza viene calcolata
    automaticamente dal raggio della scena e dall'angolo di campo.

    Args:
        scene:   La scena Blender.
        lens_mm: Lunghezza focale in mm.
        aabb:    Tupla opzionale (mins, maxs, center).

    Returns:
        Il nuovo oggetto camera, gia' collegato alla collezione della scena.
    """
    import bpy  # noqa: PLC0415
    import mathutils  # noqa: PLC0415

    mins, maxs, center = aabb if aabb is not None else _scene_aabb(scene)
    radius = max((maxs - mins).length / 2.0, 0.5)

    cam_data = bpy.data.cameras.new("NL2_Corner")
    cam_data.type = "PERSP"
    cam_data.lens = float(lens_mm)

    half_fov = max(cam_data.angle / 2.0, 1e-3)
    dist = radius / math.sin(half_fov) * 1.15

    # Direzione alta: garantisce che tutta la stanza sia visibile.
    direction = mathutils.Vector((1.0, -1.0, 1.6))
    direction.normalize()
    loc = center + direction * (dist * 1.15)

    cam = bpy.data.objects.new("NL2_Corner", cam_data)
    scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (center - loc).to_track_quat("-Z", "Y").to_euler()

    return cam


def _make_iso_camera(scene, aabb=None):
    """
    Crea una camera isometrica (proiezione ortografica obliqua 45 gradi),
    inquadrata sull'AABB fornito o sull'intera scena.

    Args:
        scene: La scena Blender.
        aabb:  Tupla opzionale (mins, maxs, center).

    Returns:
        Il nuovo oggetto camera, gia' collegato alla collezione della scena.
    """
    import bpy  # noqa: PLC0415
    import mathutils  # noqa: PLC0415

    mins, maxs, center = aabb if aabb is not None else _scene_aabb(scene)
    diag = max((maxs - mins).length, 1.0)

    cam_data = bpy.data.cameras.new("NL2_Iso")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = diag * 1.05
    cam_data.clip_start = 0.01
    cam_data.clip_end = diag * 10.0

    direction = mathutils.Vector((1.0, -1.0, 1.0))
    direction.normalize()
    loc = center + direction * (diag * 2.0)

    cam = bpy.data.objects.new("NL2_Iso", cam_data)
    scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (center - loc).to_track_quat("-Z", "Y").to_euler()

    return cam


# ---------------------------------------------------------------------------
# Funzione principale di render
# ---------------------------------------------------------------------------

def render_labeled_views(
    use_existing_camera: bool = True,
    add_top_down: bool = True,
    add_iso: bool = False,
    label_names: Optional[set] = None,
    brighten: float = 1.5,
    gamma: float = 1.6,
    lens_override: Optional[float] = None,
    auto_perspective: bool = True,
    auto_lens: float = 24.0,
    const: Constants = CONST,
) -> list[str]:
    """
    Esegue il render di una o piu' viste della scena, aggiungendo etichette,
    bussola e barra di scala, e salva i file PNG nella directory di output.

    Viste prodotte (in base ai parametri):
        1. Prospettica d'angolo: sempre generata se auto_perspective=True o
           use_existing_camera=True e una camera e' presente.
        2. Pianta top-down (ortografica): generata se add_top_down=True.
        3. Isometrica (ortografica): generata se add_iso=True.

    L'inquadratura e' basata sugli oggetti etichettati (label_names), in modo
    che mesh lontani e non etichettati non espandano inutilmente la vista.

    Le impostazioni di render (risoluzione, formato, camera) vengono salvate e
    ripristinate al termine; le camera temporanee vengono rimosse.

    Args:
        use_existing_camera: Se True, usa la camera gia' presente nella scena
                             per la vista prospettica (solo se auto_perspective=False).
        add_top_down:        Se True, genera la vista dall'alto.
        add_iso:             Se True, genera la vista isometrica.
        label_names:         Set di nomi di oggetti da etichettare. Se None,
                             etichetta tutti i MESH.
        brighten:            Fattore di luminosita' da applicare all'immagine.
        gamma:               Esponente di correzione gamma.
        lens_override:       Lunghezza focale (mm) da applicare alla camera
                             esistente per la vista prospettica.
        auto_perspective:    Se True, crea automaticamente una camera prospettica
                             d'angolo (ignora use_existing_camera).
        auto_lens:           Lunghezza focale usata per la camera auto-prospettica.
        const:               Costanti di configurazione del progetto.

    Returns:
        Lista dei percorsi assoluti dei file PNG generati.
    """
    import bpy  # noqa: PLC0415

    scene = bpy.context.scene
    out_dir = _ensure_output_dir()
    width = height = int(const.render_edge_px)

    # Seleziona gli oggetti da etichettare.
    if label_names is not None:
        mesh_objs = [o for o in scene.objects if o.type == "MESH" and o.name in label_names]
    else:
        mesh_objs = [o for o in scene.objects if o.type == "MESH"]

    # L'inquadratura e' basata sugli oggetti etichettati, non sull'intera scena.
    frame_aabb = _objects_aabb(mesh_objs) if mesh_objs else _scene_aabb(scene)
    center = frame_aabb[2]

    # Salva le impostazioni di render originali per il ripristino.
    r = scene.render
    backup = (
        r.resolution_x, r.resolution_y, r.resolution_percentage,
        r.filepath, r.image_settings.file_format, scene.camera,
    )

    r.resolution_x = width
    r.resolution_y = height
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"

    created: list[str] = []
    temp_objs: list = []

    def _shoot(cam, suffix: str) -> None:
        """
        Imposta la camera, esegue il render, applica etichette e overlay,
        e aggiunge il percorso del file alla lista dei risultati.
        """
        scene.camera = cam
        path = os.path.join(out_dir, f"{_safe(scene.name)}_{suffix}.png")
        _render_to(path)
        _draw_labels(
            path,
            _label_points(scene, cam, mesh_objs, width, height),
            brighten=brighten,
            gamma=gamma,
        )

        # La barra di scala viene aggiunta solo per le viste ortografiche.
        mpp = None
        if getattr(cam.data, "type", "") == "ORTHO":
            mpp = float(cam.data.ortho_scale) / float(width)

        axes = _axes_screen_dirs(scene, cam, width, height, center)
        _draw_overlay(path, axes, mpp)
        created.append(path)

    try:
        # 1) Vista prospettica.
        if auto_perspective:
            pcam = _make_corner_camera(scene, auto_lens, aabb=frame_aabb)
            temp_objs.append(pcam)
            _shoot(pcam, "cam")
        elif use_existing_camera:
            pcam = scene.camera or next(
                (o for o in scene.objects if o.type == "CAMERA"), None
            )
            if pcam is not None:
                cdata = pcam.data
                lens_bk = None
                if (
                    lens_override and lens_override > 0
                    and getattr(cdata, "type", "PERSP") == "PERSP"
                    and hasattr(cdata, "lens")
                ):
                    lens_bk = cdata.lens
                    cdata.lens = float(lens_override)
                try:
                    _shoot(pcam, "cam")
                finally:
                    if lens_bk is not None:
                        cdata.lens = lens_bk

        # 2) Pianta top-down.
        if add_top_down:
            tcam = _make_top_down_camera(scene, aabb=frame_aabb)
            temp_objs.append(tcam)
            _shoot(tcam, "top")

        # 3) Vista isometrica (opzionale).
        if add_iso:
            icam = _make_iso_camera(scene, aabb=frame_aabb)
            temp_objs.append(icam)
            _shoot(icam, "iso")

        logger.info("Render con etichette: %d file in %s", len(created), out_dir)
        return created

    finally:
        # Ripristina le impostazioni di render originali.
        (
            r.resolution_x, r.resolution_y, r.resolution_percentage,
            r.filepath, r.image_settings.file_format, scene.camera,
        ) = backup

        # Rimuove le camera temporanee create durante il render.
        for ob in temp_objs:
            data = ob.data
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
                bpy.data.cameras.remove(data)
            except Exception:
                pass