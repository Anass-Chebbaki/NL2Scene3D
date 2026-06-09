# nl2scene3d/core/render.py
"""
Render automatico delle viste della scena, con etichette + bussola assi + barra di scala.

  - Render OpenGL/Workbench: sempre illuminato, mai nero (luminosita' + gamma).
  - Prospettica d'angolo auto-inquadrata (mostra tutta la stanza), top-down e iso.
  - Etichette ANTI-SOVRAPPOSIZIONE (declutter) con linee di richiamo.
  - OVERLAY: bussola assi X/Y in ogni vista; barra di scala metrica SOLO nelle
    viste ortografiche (top-down/iso), dove un pixel vale sempre gli stessi metri.
    Sulla prospettica la barra non c'e' (sarebbe imprecisa): solo la bussola.

Generico: nessun nome/tipo-stanza cablato. Proiezione 3D->2D, declutter e
"nice number" della barra sono PURI. Z non viene mai toccata.
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
# Helper PURI
# ---------------------------------------------------------------------------

def _ndc_to_pixels(ndc_x: float, ndc_y: float, width: int, height: int) -> tuple[int, int]:
    return int(round(ndc_x * width)), int(round((1.0 - ndc_y) * height))


def _is_in_frame(cam_x: float, cam_y: float, cam_z: float, margin: float = 0.02) -> bool:
    return cam_z > 0.0 and -margin <= cam_x <= 1.0 + margin and -margin <= cam_y <= 1.0 + margin


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _nice_length(raw_m: float) -> float:
    """Arrotonda una lunghezza a un valore 'tondo' 1/2/5 x 10^k. PURO."""
    if raw_m <= 0:
        return 1.0
    k = math.floor(math.log10(raw_m))
    base = raw_m / (10 ** k)
    nice = 1 if base < 1.5 else (2 if base < 3.5 else 5)
    return nice * (10 ** k)


def _declutter(boxes: list, width: int, height: int, pad: int = 5, iterations: int = 150) -> list:
    """Scosta i riquadri finche' non si sovrappongono (MTV in pixel). PURO."""
    for _ in range(iterations):
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                dx = b["cx"] - a["cx"]
                dy = b["cy"] - a["cy"]
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    dy = 1.0
                ox = (a["w"] + b["w"]) / 2.0 + pad - abs(dx)
                oy = (a["h"] + b["h"]) / 2.0 + pad - abs(dy)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        s = ox / 2.0 * (1.0 if dx >= 0 else -1.0)
                        a["cx"] -= s; b["cx"] += s
                    else:
                        s = oy / 2.0 * (1.0 if dy >= 0 else -1.0)
                        a["cy"] -= s; b["cy"] += s
                    moved = True
        for bx in boxes:
            hw, hh = bx["w"] / 2.0, bx["h"] / 2.0
            bx["cx"] = min(max(bx["cx"], hw), width - hw)
            bx["cy"] = min(max(bx["cy"], hh), height - hh)
        if not moved:
            break
    return boxes


# ---------------------------------------------------------------------------
# Disegno etichette (Pillow): schiarisce, declutter, richiami
# ---------------------------------------------------------------------------

def _gutter_layout(boxes, W, H, gap=10, edge=8):
    """
    PURO. Assegna ogni etichetta al bordo (L/R/T/B) piu' vicino al suo oggetto,
    calcola i margini (cornice) necessari e impacchetta le etichette lungo ogni
    bordo senza sovrapporle. Ritorna (ml, mr, mt, mb, Wp, Hp); scrive cx, cy
    (centro box, in coordinate del canvas con cornice) e 'border' in ogni box.
    """
    for b in boxes:
        dl, dr, dt, db = b["ax"], W - b["ax"], b["ay"], H - b["ay"]
        m = min(dl, dr, dt, db)
        b["border"] = "L" if m == dl else "R" if m == dr else "T" if m == dt else "B"

    L = [b for b in boxes if b["border"] == "L"]
    R = [b for b in boxes if b["border"] == "R"]
    T = [b for b in boxes if b["border"] == "T"]
    B = [b for b in boxes if b["border"] == "B"]

    ml = (max(b["w"] for b in L) + 2 * gap) if L else gap
    mr = (max(b["w"] for b in R) + 2 * gap) if R else gap
    mt = (max(b["h"] for b in T) + 2 * gap) if T else gap
    mb = (max(b["h"] for b in B) + 2 * gap) if B else gap
    Wp, Hp = W + ml + mr, H + mt + mb

    def pack(items, key, dim, lo, hi):
        items = sorted(items, key=key)
        total = sum(b[dim] for b in items) + gap * max(0, len(items) - 1)
        cur = lo + max(0.0, (hi - lo - total)) / 2.0
        for b in items:
            b["_pos"] = cur + b[dim] / 2.0
            cur += b[dim] + gap

    pack(L, lambda b: b["ay"], "h", edge, Hp - edge)
    for b in L:
        b["cx"] = ml - gap - b["w"] / 2.0; b["cy"] = b["_pos"]
    pack(R, lambda b: b["ay"], "h", edge, Hp - edge)
    for b in R:
        b["cx"] = Wp - mr + gap + b["w"] / 2.0; b["cy"] = b["_pos"]
    pack(T, lambda b: b["ax"], "w", edge, Wp - edge)
    for b in T:
        b["cy"] = mt - gap - b["h"] / 2.0; b["cx"] = b["_pos"]
    pack(B, lambda b: b["ax"], "w", edge, Wp - edge)
    for b in B:
        b["cy"] = Hp - mb + gap + b["h"] / 2.0; b["cx"] = b["_pos"]
    return ml, mr, mt, mb, Wp, Hp


def _draw_labels(image_path, points, font_size=18, brighten=1.5, gamma=1.6) -> bool:
    """
    Etichette nella CORNICE attorno alla scena (mai sopra la scena): la scena
    resta interamente visibile, ogni etichetta sta nel margine del bordo piu'
    vicino al suo oggetto ed e' collegata da una linea di richiamo.
    """
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFont  # type: ignore
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # tollera PNG non completati al 100%
    except ImportError:
        if points:
            sidecar = os.path.splitext(image_path)[0] + ".labels.json"
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump([{"name": n, "x": x, "y": y} for n, x, y in points], f, indent=2)
            logger.warning("Pillow non disponibile: etichette non disegnate, scritto %s", sidecar)
        return False

    base = Image.open(image_path).convert("RGB")
    if brighten and abs(brighten - 1.0) > 1e-3:
        base = ImageEnhance.Brightness(base).enhance(brighten)
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
    boxes = []
    for name, x, y in points:
        tb = measure.textbbox((0, 0), name, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        boxes.append({"name": name, "ax": float(x), "ay": float(y),
                      "w": tw + 10, "h": th + 8, "tw": tw, "th": th, "ox": tb[0], "oy": tb[1]})

    ml, mr, mt, mb, Wp, Hp = _gutter_layout(boxes, W, H)

    # canvas con cornice: sfondo grigio scuro neutro, scena incollata al centro
    canvas = Image.new("RGB", (int(Wp), int(Hp)), (40, 40, 40))
    canvas.paste(base, (int(ml), int(mt)))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for b in boxes:
        ax, ay = b["ax"] + ml, b["ay"] + mt          # ancora nello spazio del canvas
        cx, cy = b["cx"], b["cy"]
        # punto di partenza della linea: bordo del box rivolto verso la scena
        if b["border"] == "L":   sx, sy = cx + b["w"] / 2.0, cy
        elif b["border"] == "R": sx, sy = cx - b["w"] / 2.0, cy
        elif b["border"] == "T": sx, sy = cx, cy + b["h"] / 2.0
        else:                    sx, sy = cx, cy - b["h"] / 2.0
        draw.line((sx, sy, ax, ay), fill=(255, 255, 255, 230), width=1)
        draw.ellipse((ax - 2, ay - 2, ax + 2, ay + 2), fill=(255, 255, 255, 255))
        # etichetta: riquadro NERO OPACO + testo bianco
        x0, y0 = cx - b["w"] / 2.0, cy - b["h"] / 2.0
        draw.rectangle((x0, y0, x0 + b["w"], y0 + b["h"]), fill=(0, 0, 0, 255))
        tx = cx - b["tw"] / 2.0 - b["ox"]
        ty = cy - b["th"] / 2.0 - b["oy"]
        draw.text((tx, ty), b["name"], font=font, fill=(255, 255, 255, 255))

    Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB").save(image_path)
    return True


def _draw_overlay(image_path, axes_dirs, meters_per_pixel=None) -> None:
    """Disegna bussola assi (X rosso, Y verde) e, se ortho, la barra di scala."""
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
        font_axis = ImageFont.truetype("DejaVuSans.ttf", 24)   # bussola piu' grande
    except OSError:
        font = ImageFont.load_default()
        font_axis = font

    # --- bussola assi (in alto a destra), ingrandita ---
    if axes_dirs:
        (xdx, xdy), (ydx, ydy) = axes_dirs
        ax, ay, L = W - 92, 92, 56
        draw.line((ax, ay, ax + xdx * L, ay + xdy * L), fill=(235, 70, 70, 255), width=5)
        draw.text((ax + xdx * (L + 14) - 7, ay + xdy * (L + 14) - 12), "X", font=font_axis, fill=(235, 70, 70, 255))
        draw.line((ax, ay, ax + ydx * L, ay + ydy * L), fill=(70, 200, 70, 255), width=5)
        draw.text((ax + ydx * (L + 14) - 7, ay + ydy * (L + 14) - 12), "Y", font=font_axis, fill=(70, 200, 70, 255))
        draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill=(255, 255, 255, 255))

    # --- barra di scala (in basso a sinistra), solo viste ortografiche ---
    if meters_per_pixel and meters_per_pixel > 0:
        bar_m = _nice_length(140 * meters_per_pixel)
        bar_px = int(round(bar_m / meters_per_pixel))
        x0, y0 = 30, H - 36
        draw.rectangle((x0 - 7, y0 - 22, x0 + bar_px + 7, y0 + 10), fill=(0, 0, 0, 150))
        draw.line((x0, y0, x0 + bar_px, y0), fill=(255, 255, 255, 255), width=3)
        draw.line((x0, y0 - 6, x0, y0 + 6), fill=(255, 255, 255, 255), width=3)
        draw.line((x0 + bar_px, y0 - 6, x0 + bar_px, y0 + 6), fill=(255, 255, 255, 255), width=3)
        draw.text((x0, y0 - 20), f"{bar_m:g} m", font=font, fill=(255, 255, 255, 255))

    Image.alpha_composite(img, overlay).convert("RGB").save(image_path)


# ---------------------------------------------------------------------------
# Funzioni bpy-facing
# ---------------------------------------------------------------------------

def _ensure_output_dir(subdir: str = "nl2_renders") -> str:
    import bpy  # noqa: PLC0415
    base = bpy.path.abspath("//" + subdir + "/") if bpy.data.filepath \
        else os.path.join(tempfile.gettempdir(), subdir)
    os.makedirs(base, exist_ok=True)
    return base


def _objects_aabb(objs):
    """AABB unione (mondo) dei soli MESH nella lista. (mins, maxs, center)."""
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


def _scene_aabb(scene, objs=None):
    """
    AABB di inquadratura. Se 'objs' e' fornito, inquadra SOLO quelli (gli oggetti
    etichettati): cosi' mesh sparsi lontani (es. una collezione 'luces') non
    gonfiano la vista. Altrimenti usa tutti i MESH della scena.
    """
    if objs:
        return _objects_aabb(objs)
    return _objects_aabb(list(scene.objects))


def _axes_screen_dirs(scene, cam, width, height, center):
    """Direzioni schermo (unitarie) degli assi mondo +X e +Y, viste da 'cam'."""
    import mathutils  # noqa: PLC0415
    from bpy_extras.object_utils import world_to_camera_view  # noqa: PLC0415

    C = mathutils.Vector(center)
    k = 0.5

    def to_px(v):
        co = world_to_camera_view(scene, cam, v)
        return _ndc_to_pixels(co.x, co.y, width, height)

    def unit(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = (dx * dx + dy * dy) ** 0.5
        return (1.0, 0.0) if n < 1e-6 else (dx / n, dy / n)

    p0 = to_px(C)
    return (unit(p0, to_px(C + mathutils.Vector((k, 0, 0)))),
            unit(p0, to_px(C + mathutils.Vector((0, k, 0)))))


def _label_points(scene, cam, mesh_objs, width: int, height: int):
    import mathutils  # noqa: PLC0415
    from bpy_extras.object_utils import world_to_camera_view  # noqa: PLC0415

    pts = []
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


def _make_top_down_camera(scene, aabb=None):
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
    import bpy        # noqa: PLC0415
    import mathutils  # noqa: PLC0415

    mins, maxs, center = aabb if aabb is not None else _scene_aabb(scene)
    radius = max((maxs - mins).length / 2.0, 0.5)

    cam_data = bpy.data.cameras.new("NL2_Corner")
    cam_data.type = "PERSP"
    cam_data.lens = float(lens_mm)
    half_fov = max(cam_data.angle / 2.0, 1e-3)
    dist = radius / math.sin(half_fov) * 1.15

    direction = mathutils.Vector((1.0, -1.0, 1.6))   # alta: vede tutta la stanza
    direction.normalize()
    loc = center + direction * (dist * 1.15)

    cam = bpy.data.objects.new("NL2_Corner", cam_data)
    scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (center - loc).to_track_quat("-Z", "Y").to_euler()
    return cam


def _make_iso_camera(scene, aabb=None):
    import bpy        # noqa: PLC0415
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
    import bpy  # noqa: PLC0415

    scene   = bpy.context.scene
    out_dir = _ensure_output_dir()
    width   = height = int(const.render_edge_px)

    if label_names is not None:
        mesh_objs = [o for o in scene.objects if o.type == "MESH" and o.name in label_names]
    else:
        mesh_objs = [o for o in scene.objects if o.type == "MESH"]

    # Inquadratura basata sugli oggetti ETICHETTATI (non tutta la scena): cosi'
    # mesh lontani non etichettati (es. collezione 'luces') non gonfiano la vista.
    frame_aabb = _objects_aabb(mesh_objs) if mesh_objs else _scene_aabb(scene)
    center = frame_aabb[2]

    r = scene.render
    backup = (
        r.resolution_x, r.resolution_y, r.resolution_percentage,
        r.filepath, r.image_settings.file_format, scene.camera,
    )
    r.resolution_x = width
    r.resolution_y = height
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"

    created = []
    temp_objs = []

    def _shoot(cam, suffix):
        scene.camera = cam
        path = os.path.join(out_dir, f"{_safe(scene.name)}_{suffix}.png")
        _render_to(path)
        _draw_labels(path, _label_points(scene, cam, mesh_objs, width, height),
                     brighten=brighten, gamma=gamma)
        # overlay: barra di scala solo se ortografica (mpp costante), bussola sempre
        mpp = None
        if getattr(cam.data, "type", "") == "ORTHO":
            mpp = float(cam.data.ortho_scale) / float(width)
        axes = _axes_screen_dirs(scene, cam, width, height, center)
        _draw_overlay(path, axes, mpp)
        created.append(path)

    try:
        # 1) prospettica
        if auto_perspective:
            pcam = _make_corner_camera(scene, auto_lens, aabb=frame_aabb)
            temp_objs.append(pcam)
            _shoot(pcam, "cam")
        elif use_existing_camera:
            pcam = scene.camera or next((o for o in scene.objects if o.type == "CAMERA"), None)
            if pcam is not None:
                cdata = pcam.data
                lens_bk = None
                if lens_override and lens_override > 0 and getattr(cdata, "type", "PERSP") == "PERSP" \
                        and hasattr(cdata, "lens"):
                    lens_bk = cdata.lens
                    cdata.lens = float(lens_override)
                try:
                    _shoot(pcam, "cam")
                finally:
                    if lens_bk is not None:
                        cdata.lens = lens_bk

        # 2) pianta top-down
        if add_top_down:
            tcam = _make_top_down_camera(scene, aabb=frame_aabb)
            temp_objs.append(tcam)
            _shoot(tcam, "top")

        # 3) isometrica (opzionale)
        if add_iso:
            icam = _make_iso_camera(scene, aabb=frame_aabb)
            temp_objs.append(icam)
            _shoot(icam, "iso")

        logger.info("Render con etichette: %d file in %s", len(created), out_dir)
        return created

    finally:
        (r.resolution_x, r.resolution_y, r.resolution_percentage,
         r.filepath, r.image_settings.file_format, scene.camera) = backup
        for ob in temp_objs:
            data = ob.data
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
                bpy.data.cameras.remove(data)
            except Exception:
                pass