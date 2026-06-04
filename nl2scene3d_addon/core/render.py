# nl2scene3d/core/render.py
"""
Render automatico delle viste della scena, con le etichette (nome oggetto)
sovrapposte. E' lo "Step 4": generare le immagini da allegare all'LLM senza
doverle fare a mano.

Modulo bpy-facing (come scene_io): la logica di rendering vive qui, mentre la
matematica di proiezione 3D -> 2D, che e' PURA, sta in _ndc_to_pixels /
_is_in_frame e si puo' testare senza Blender (come il resto del core).

Le etichette vengono disegnate SULL'immagine renderizzata (con Pillow), non con
text-object 3D: cosi' non si sporca la scena e non resta nulla da ripulire. Se
Pillow non e' installato nel Python di Blender, il render viene comunque salvato
SENZA etichette e accanto viene scritto un sidecar '.labels.json' con le
posizioni in pixel: la funzione non fallisce mai per colpa di una dipendenza.

Z non viene mai toccata: il render fotografa la scena com'e'.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from .settings import CONST, Constants, STRUCTURAL_PATTERNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper PURI (testabili senza Blender)
# ---------------------------------------------------------------------------

def _ndc_to_pixels(ndc_x: float, ndc_y: float, width: int, height: int) -> tuple[int, int]:
    """
    Converte coordinate normalizzate camera-view (0..1, origine in BASSO a
    sinistra, com'e' il valore di world_to_camera_view) in pixel immagine
    (origine in ALTO a sinistra). Puro: nessun bpy.
    """
    px = int(round(ndc_x * width))
    py = int(round((1.0 - ndc_y) * height))
    return px, py


def _is_in_frame(cam_x: float, cam_y: float, cam_z: float, margin: float = 0.02) -> bool:
    """True se il punto e' DAVANTI alla camera (z>0) e dentro il frame (+margine)."""
    return cam_z > 0.0 and -margin <= cam_x <= 1.0 + margin and -margin <= cam_y <= 1.0 + margin


def _safe(name: str) -> str:
    """Rende un nome scena/oggetto sicuro per un filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _looks_movable(obj_name: str) -> bool:
    """Stima rapida 'mobile' per nome (esclude gli strutturali). Puro."""
    low = obj_name.lower()
    return not any(p in low for p in STRUCTURAL_PATTERNS)


# ---------------------------------------------------------------------------
# Disegno etichette (Pillow, con fallback)
# ---------------------------------------------------------------------------

def _draw_labels(image_path: str, points: list[tuple[str, int, int]], font_size: int = 18, brighten: float = 1.5,
    gamma: float = 1.6,) -> bool:
    """
    Disegna le etichette sull'immagine gia' salvata. Ritorna True se ha usato
    Pillow, False se Pillow non c'e' (in quel caso scrive un sidecar .labels.json
    e non disegna nulla).
    """
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFont 
    except ImportError:
        sidecar = os.path.splitext(image_path)[0] + ".labels.json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump([{"name": n, "x": x, "y": y} for n, x, y in points], f, indent=2)
        logger.warning("Pillow non disponibile: etichette non disegnate, scritto %s", sidecar)
        return False

    img = Image.open(image_path).convert("RGBA")

    # 1) schiarisci: brightness lineare + un gamma che ALZA le ombre, cosi' anche
    #    i materiali scuri (mobili neri) diventano leggibili senza bruciare le luci.
    rgb = img.convert("RGB")
    if brighten and abs(brighten - 1.0) > 1e-3:
        rgb = ImageEnhance.Brightness(rgb).enhance(brighten)
    if gamma and abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / gamma
        lut = [min(255, int(round(255.0 * ((i / 255.0) ** inv)))) for i in range(256)]
        rgb = rgb.point(lut * 3)
    img = rgb.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for name, x, y in points:
        tb       = draw.textbbox((0, 0), name, font=font)
        tw, th   = tb[2] - tb[0], tb[3] - tb[1]
        bx, by   = x - tw // 2, y - th // 2
        draw.rectangle((bx - 3, by - 3, bx + tw + 3, by + th + 3), fill=(0, 0, 0, 160))
        draw.text((bx, by), name, font=font, fill=(255, 255, 255, 255))

    Image.alpha_composite(img, overlay).convert("RGB").save(image_path)
    return True


# ---------------------------------------------------------------------------
# Funzioni bpy-facing (import di bpy SOLO dentro le funzioni, come scene_io)
# ---------------------------------------------------------------------------

def _ensure_output_dir(subdir: str = "nl2_renders") -> str:
    """Cartella di output: accanto al .blend se salvato, altrimenti in temp."""
    import bpy  # noqa: PLC0415
    base = bpy.path.abspath("//" + subdir + "/") if bpy.data.filepath \
        else os.path.join(tempfile.gettempdir(), subdir)
    os.makedirs(base, exist_ok=True)
    return base


def _label_points(scene, cam, mesh_objs, width: int, height: int) -> list[tuple[str, int, int]]:
    """
    Per ogni oggetto, se il suo centro e' visibile in camera, ritorna
    (nome, px, py). Usa world_to_camera_view (gestisce sia PERSP che ORTHO).
    """
    import mathutils  # noqa: PLC0415
    from bpy_extras.object_utils import world_to_camera_view  # noqa: PLC0415

    pts: list[tuple[str, int, int]] = []
    for ob in mesh_objs:
        local_center = sum((mathutils.Vector(c) for c in ob.bound_box), mathutils.Vector()) / 8.0
        world_center = ob.matrix_world @ local_center
        co = world_to_camera_view(scene, cam, world_center)
        if not _is_in_frame(co.x, co.y, co.z):
            continue
        px, py = _ndc_to_pixels(co.x, co.y, width, height)
        pts.append((ob.name, px, py))
    return pts


def _render_to(path: str) -> None:
    """
    Render della vista corrente con il motore OpenGL/Workbench (come il viewport
    in 'solid'): SEMPRE illuminato dalle studio-light, quindi non dipende dalle
    luci della scena ne' dal mondo. Per far capire il layout all'LLM e' anche piu'
    pulito e veloce. Se l'OpenGL non fosse disponibile, ripiega sul render normale.
    """
    import bpy  # noqa: PLC0415
    scene = bpy.context.scene
    scene.render.filepath = path

    # Workbench: studio-light sempre accesa + texture a colori.
    try:
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
    except Exception:
        pass

    try:
        bpy.ops.render.opengl(write_still=True, view_context=False)
    except RuntimeError:
        bpy.ops.render.render(write_still=True)

def _make_top_down_camera(scene):
    """Crea una camera ortografica dall'alto, inquadrando tutti i MESH. Da rimuovere a fine render."""
    import bpy        # noqa: PLC0415
    import mathutils  # noqa: PLC0415

    cam_data = bpy.data.cameras.new("NL2_TopDown")
    cam_data.type = "ORTHO"

    big = 1.0e9
    mins = mathutils.Vector((big, big, big))
    maxs = mathutils.Vector((-big, -big, -big))
    for o in scene.objects:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(c)
            mins = mathutils.Vector((min(mins.x, w.x), min(mins.y, w.y), min(mins.z, w.z)))
            maxs = mathutils.Vector((max(maxs.x, w.x), max(maxs.y, w.y), max(maxs.z, w.z)))

    center = (mins + maxs) / 2.0
    span   = max(maxs.x - mins.x, maxs.y - mins.y, 1.0) * 1.10
    cam_data.ortho_scale = span

    cam = bpy.data.objects.new("NL2_TopDown", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (center.x, center.y, maxs.z + 5.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)  # la camera guarda lungo -Z: dritto in giu'
    return cam


def render_labeled_views(
    use_existing_camera: bool = True,
    add_top_down: bool = True,
    label_names: Optional[set] = None,
    brighten: float = 1.5,
    gamma: float = 1.6,
    const: Constants = CONST,
) -> list[str]:
    """
    Renderizza la scena e salva PNG con i nomi degli oggetti sovrapposti.
 
    Parametri:
        use_existing_camera: usa scene.camera (o la prima Camera trovata).
        add_top_down:        aggiunge una vista ortografica dall'alto (camera
                             temporanea, creata e rimossa al volo).
        label_names:         insieme dei nomi da etichettare. Di norma sono i
                             nomi che escono da reorganizer.build_request (root
                             mobili + ostacoli fissi): cosi' l'immagine mostra
                             ESATTAMENTE cio' che e' nel JSON, senza figli ne'
                             stanza. Se None, etichetta tutti i MESH.
 
    Ritorna la lista dei file PNG generati (puo' essere vuota se non c'e' camera).
    """
    import bpy  # noqa: PLC0415
 
    scene   = bpy.context.scene
    out_dir = _ensure_output_dir()
    width   = height = int(const.render_edge_px)
 
    if label_names is not None:
        mesh_objs = [o for o in scene.objects if o.type == "MESH" and o.name in label_names]
    else:
        mesh_objs = [o for o in scene.objects if o.type == "MESH"]
 
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
    try:
        # 1) vista dalla camera esistente
        cam = scene.camera or next((o for o in scene.objects if o.type == "CAMERA"), None)
        if use_existing_camera and cam is not None:
            scene.camera = cam
            path = os.path.join(out_dir, f"{_safe(scene.name)}_cam.png")
            _render_to(path)
            _draw_labels(path, _label_points(scene, cam, mesh_objs, width, height), brighten=brighten, gamma=gamma)
            created.append(path)
 
        # 2) vista ortografica dall'alto (camera temporanea)
        if add_top_down:
            tcam = _make_top_down_camera(scene)
            temp_objs.append(tcam)
            scene.camera = tcam
            path = os.path.join(out_dir, f"{_safe(scene.name)}_top.png")
            _render_to(path)
            _draw_labels(path, _label_points(scene, tcam, mesh_objs, width, height), brighten=brighten, gamma=gamma)
            created.append(path)
 
        logger.info("Render con etichette: %d file in %s", len(created), out_dir)
        return created
 
    finally:
        # ripristino impostazioni render + rimozione camere temporanee
        (r.resolution_x, r.resolution_y, r.resolution_percentage,
         r.filepath, r.image_settings.file_format, scene.camera) = backup
        for ob in temp_objs:
            data = ob.data
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
                bpy.data.cameras.remove(data)
            except Exception:
                pass