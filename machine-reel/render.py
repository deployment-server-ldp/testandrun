#!/usr/bin/env python3
"""Cigarette Making Machine - exploded view reel (1080x1920, 25 s, 30 fps).

Everything is a separate layer: background, machine photo cut-out, one RGBA layer per
component group (layers/parts/*.png, cut from the infographic), callouts, titles, camera.
Layout / labels / timing are data: edit parts.json and the TIMING block below, then

    python3 render.py                 # full render -> out/cigarette_machine_exploded_view.mp4
    python3 render.py --still 12.5    # one frame  -> build/still_12.50.png
"""
import json, math, os, subprocess, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

R = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, DUR = 1080, 1920, 30, 25.0
BLUE = np.array([255, 170, 30], np.float32)      # BGR electric blue
CYAN = np.array([255, 225, 130], np.float32)
FD = "/usr/share/fonts/opentype/inter/"
F_TITLE, F_LABEL, F_SUB = FD + "Inter-ExtraBoldItalic.otf", FD + "Inter-SemiBold.otf", FD + "Inter-SemiBold.otf"

# ------------------------------------------------------------------ TIMING
T_EXPLODE, EXPLODE_DUR = 4.15, 2.4       # scene 2
T_LABELS, LABEL_STEP = 10.1, 0.25       # scene 3
T_LABELS_OUT = 15.6
T_REASM, REASM_DUR = 16.25, 2.2          # scene 4
T_FINAL = 22.0                           # scene 5
ORDER = ["frame", "tobacco", "paper", "cover_r", "cover_l", "local_panel", "glue", "panel", "motor",
         "pneumatic", "cutting", "tray", "aux_motor", "gearbox", "pulleys", "gears", "belts", "rollers",
         "valves", "sensors", "bearings"]
LABEL_ORDER = ["frame", "motor", "aux_motor", "gearbox", "gears", "pulleys", "belts", "rollers", "cutting",
               "tobacco", "paper", "glue", "pneumatic", "valves", "panel", "sensors", "bearings", "cover_r", "tray"]
CAM_KEYS = [(0, 1.0), (4.0, 1.10), (7.2, 1.0), (10, 1.0), (16, 1.025), (21.6, 1.12), (25, 1.0)]
PAN_KEYS = [(0, 95), (4.0, 95), (7.2, 0), (16.5, 0), (21.0, 95), (25, 95)]   # camera drop so the hero machine sits lower under the titles
# machine photo -> world mapping
PS, PHOTO_C, WORLD_C = 0.60, (900, 705), (540, 1010)

# ------------------------------------------------------------------ helpers
clamp = lambda v, a=0.0, b=1.0: max(a, min(b, v))
def sstep(a, b, v):
    v = clamp((v - a) / (b - a)); return v * v * (3 - 2 * v)
out_cubic = lambda v: 1 - (1 - clamp(v)) ** 3
def out_back(v, s=1.1):
    v = clamp(v) - 1; return 1 + v * v * ((s + 1) * v + s)
def io_cubic(v):
    v = clamp(v); return 4 * v ** 3 if v < .5 else 1 - (-2 * v + 2) ** 3 / 2
def p2w(px, py):
    return ((px - PHOTO_C[0]) * PS + WORLD_C[0], (py - PHOTO_C[1]) * PS + WORLD_C[1])

def cam_zoom(t):
    for (t0, z0), (t1, z1) in zip(CAM_KEYS, CAM_KEYS[1:]):
        if t <= t1:
            return z0 + (z1 - z0) * (0.5 - 0.5 * math.cos(math.pi * clamp((t - t0) / (t1 - t0))))
    return CAM_KEYS[-1][1]
def key(keys, t):
    for (t0, z0), (t1, z1) in zip(keys, keys[1:]):
        if t <= t1:
            return z0 + (z1 - z0) * (0.5 - 0.5 * math.cos(math.pi * clamp((t - t0) / (t1 - t0))))
    return keys[-1][1]
PAN = [0.0]
def cam(x, y, z, depth=1.0):
    z = 1 + (z - 1) * depth
    return (x - 540) * z + 540, (y - 985) * z + 985 + PAN[0], z

def premult(rgba_u8):
    a = rgba_u8[..., 3:4].astype(np.float32) / 255
    return np.dstack([rgba_u8[..., :3].astype(np.float32) * a, a])

def comp(cv, sp, x, y, alpha=1.0, add=False):
    """composite premultiplied float sprite (h,w,4) with top-left at integer x,y"""
    h, w = sp.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.003: return
    s = sp[y0 - y:y1 - y, x0 - x:x1 - x]
    d = cv[y0:y1, x0:x1]
    if add: d += s[..., :3] * alpha
    else:
        d *= 1 - s[..., 3:4] * alpha; d += s[..., :3] * alpha

def draw_sprite(cv, sp, cx, cy, f, alpha=1.0, add=False, gain=1.0):
    """draw premultiplied sprite scaled by f, centred on sub-pixel cx,cy"""
    h, w = sp.shape[:2]
    dw, dh = w * f, h * f
    x, y = cx - dw / 2, cy - dh / 2
    ix, iy = math.floor(x), math.floor(y)
    ow, oh = int(math.ceil(dw)) + 2, int(math.ceil(dh)) + 2
    if ix > W or iy > H or ix + ow < 0 or iy + oh < 0: return
    M = np.float32([[f, 0, x - ix + (f - 1) / 2], [0, f, y - iy + (f - 1) / 2]])
    o = cv2.warpAffine(sp, M, (ow, oh), flags=cv2.INTER_AREA if f < 0.6 else cv2.INTER_LINEAR)
    if gain != 1.0: o = o.copy(); o[..., :3] *= gain
    comp(cv, o, ix, iy, alpha, add)

def text_layer(text, path, size, color=(255, 255, 255), tracking=0, glow=0, max_w=None):
    """render text to a premultiplied BGRA float layer (optionally with a blue glow under it)"""
    font = ImageFont.truetype(path, size)
    def width(f): return sum(f.getlength(c) + tracking for c in text) - tracking
    if max_w and width(font) > max_w:
        size = int(size * max_w / width(font)); font = ImageFont.truetype(path, size)
    pad = 30 + glow * 3
    w, h = int(width(font)) + 2 * pad, int(size * 1.35) + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = pad
    for c in text:
        d.text((x, pad), c, font=font, fill=color + (255,)); x += font.getlength(c) + tracking
    a = np.array(im)[..., ::-1][..., [1, 2, 3, 0]] if False else np.array(im)
    bgra = a[..., [2, 1, 0, 3]]
    lay = premult(bgra)
    if glow:
        g = cv2.GaussianBlur(lay[..., 3], (0, 0), glow)
        gl = np.dstack([g[..., None] * BLUE * 0.9, g * 0.0])       # additive-ish glow (no alpha)
        lay = np.dstack([gl[..., :3] * (1 - lay[..., 3:4]) + lay[..., :3], lay[..., 3]])
    return lay

def pill_layer(text):
    S = 2
    font = ImageFont.truetype(F_LABEL, 25 * S)
    tw = int(font.getlength(text)); px, py = 15 * S, 9 * S
    w, h = tw + 2 * px + 14 * S, 25 * S + 2 * py + 6 * S
    im = Image.new("RGBA", (w + 8 * S, h + 8 * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    o = 4 * S
    d.rounded_rectangle([o, o, o + w, o + h], radius=9 * S, fill=(6, 16, 36, 225), outline=(40, 175, 255, 255), width=int(1.5 * S))
    d.rectangle([o + 9 * S, o + h / 2 - 7 * S, o + 12 * S, o + h / 2 + 7 * S], fill=(60, 190, 255, 255))
    d.text((o + px + 9 * S, o + py - 1 * S), text, font=font, fill=(240, 248, 255, 255))
    im = im.resize((im.width // S, im.height // S), Image.LANCZOS)
    return premult(np.array(im)[..., [2, 1, 0, 3]])

# ------------------------------------------------------------------ assets
def load_assets():
    A = {}
    cfg = json.load(open(f"{R}/parts.json"))["parts"]
    parts = {}
    for i, p in enumerate(cfg):
        im = cv2.imread(f"{R}/layers/parts/{p['id']}.png", cv2.IMREAD_UNCHANGED)
        sp = premult(im)                                            # 2x source resolution
        halo = cv2.GaussianBlur(cv2.copyMakeBorder(sp[..., 3], 40, 40, 40, 40, cv2.BORDER_CONSTANT), (0, 0), 14)
        halo = cv2.resize(halo, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        p["sp"] = sp
        p["sp_half"] = cv2.resize(sp, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        p["halo"] = np.dstack([halo[..., None] * BLUE, halo])
        p["home_w"] = p2w(*p["home"])
        p["delay"] = ORDER.index(p["id"]) * 0.09
        p["phase"] = i * 1.7
        p["depth"] = 0.85 + 0.3 * ((i * 37) % 10) / 10
        if "label" in p:
            p["pill"] = pill_layer(p["label"]); p["li"] = LABEL_ORDER.index(p["id"])
        parts[p["id"]] = p
    A["parts"] = [parts[k] for k in ORDER]
    # machine photo cut-out (hero) + floor reflection + halo
    cut = cv2.imread(f"{R}/layers/machine_wide_cut.png", cv2.IMREAD_UNCHANGED)
    a = cut[..., 3]
    yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
    a[((yy < 396) & (xx > 1204)) | (xx > 1684) | (yy < 380)] = 0       # remove the building's steel brace
    x0, y0, x1, y1 = 90, 370, 1710, 1040
    hero = premult(cut[y0:y1, x0:x1]); hs = 0.75
    hero = cv2.resize(hero, None, fx=hs, fy=hs, interpolation=cv2.INTER_AREA)
    refl = hero[::-1].copy()
    ramp = np.clip(1 - np.arange(refl.shape[0]) / 150.0, 0, 1)[:, None, None] ** 1.5 * 0.22
    refl = cv2.GaussianBlur(refl, (0, 0), 2.0) * ramp
    halo = cv2.GaussianBlur(cv2.copyMakeBorder(hero[..., 3], 60, 60, 60, 60, cv2.BORDER_CONSTANT), (0, 0), 26)
    A["hero"], A["hero_s"], A["refl"] = hero, hs, refl[:170]
    A["hero_halo"] = np.dstack([halo[..., None] * BLUE * 0.55, halo * 0])
    A["hero_c"] = p2w((x0 + x1) / 2, (y0 + y1) / 2)
    A["hero_h"] = (y1 - y0) * PS
    # background
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = (yy / H)[..., None]
    bg = np.array([30, 13, 5], np.float32) * (1 - g) + np.array([14, 6, 3], np.float32) * g
    r = np.sqrt(((xx - 540) / 620) ** 2 + ((yy - 1010) / 760) ** 2)
    bg += np.clip(1 - r, 0, 1)[..., None] ** 2 * np.array([70, 30, 8], np.float32)
    A["bg"] = bg
    GW, GH = 1400, 2400
    grid = np.zeros((GH, GW, 3), np.float32)
    for x in range(0, GW, 60): grid[:, x] += BLUE * (0.10 if x % 240 == 0 else 0.04)
    for y in range(0, GH, 60): grid[y, :] += BLUE * (0.10 if y % 240 == 0 else 0.04)
    A["grid"] = grid
    A["vig"] = (1 - 0.55 * np.clip(np.sqrt(((xx - 540) / 800) ** 2 + ((yy - 960) / 1300) ** 2) - 0.45, 0, 1) ** 1.5)[..., None]
    rng = np.random.default_rng(7)
    A["dust"] = rng.random((70, 5))
    A["grain"] = [rng.normal(0, 1.3, (H // 2, W // 2)).astype(np.float32) for _ in range(8)]
    # titles
    T = {}
    T["t1"] = text_layer("CIGARETTE MAKING", F_TITLE, 98, glow=9, max_w=930)
    T["t2"] = text_layer("MACHINE", F_TITLE, 98, glow=9, max_w=930)
    T["t3"] = text_layer("ENGINEERING EXPLODED VIEW", F_SUB, 33, color=(110, 205, 255), tracking=9)
    T["c1"] = text_layer("PRECISION IN", F_TITLE, 96, glow=9, max_w=930)
    T["c2"] = text_layer("EVERY COMPONENT", F_TITLE, 96, glow=9, max_w=930)
    T["c3"] = text_layer("ENGINEERED FOR INDUSTRIAL PERFORMANCE", F_SUB, 28, color=(110, 205, 255), tracking=6)
    T["h"] = text_layer("CIGARETTE MAKING MACHINE", F_TITLE, 44, glow=5)
    for k, s in [("h2", "EXPLODED VIEW"), ("h3", "SPARE PARTS IDENTIFICATION"), ("h4", "REASSEMBLY")]:
        T[k] = text_layer(s, F_SUB, 27, color=(110, 205, 255), tracking=8)
    T["ok"] = text_layer("ASSEMBLY ALIGNED", FD + "Inter-SemiBold.otf", 22, color=(110, 205, 255), tracking=6)
    for i, s in enumerate(["ORIGINAL", "EXPLODE", "IDENTIFY", "REASSEMBLE", "FINAL"]):
        T[f"s{i}on"] = text_layer(f"0{i+1}  {s}", F_SUB, 19, color=(255, 255, 255), tracking=2)
        T[f"s{i}off"] = text_layer(f"0{i+1}  {s}", F_SUB, 19, color=(95, 125, 165), tracking=2)
    A["T"] = T
    return A

# ------------------------------------------------------------------ per-frame state
def part_e(p, t):
    """0 = assembled (home on the machine), 1 = fully exploded"""
    if t < T_REASM:
        d = EXPLODE_DUR * (1.25 if p["id"] == "frame" else 1)
        return out_back((t - T_EXPLODE - p["delay"]) / d, 0.9)
    if p["id"] == "frame":
        return 1 - io_cubic((t - T_REASM - 0.2) / 3.7)
    return 1 - io_cubic((t - T_REASM - (1.8 - p["delay"]) * 0.9) / REASM_DUR)

def part_xform(p, t):
    e = part_e(p, t); ec = clamp(e)
    hx, hy = p["home_w"]; ex, ey = p["pos"]
    x, y = hx + (ex - hx) * e, hy + (ey - hy) * e
    s = (p["home_scale"] + (p["scale"] - p["home_scale"]) * e) * (1 + 0.07 * math.sin(math.pi * ec))
    y += 3.2 * math.sin(t * 1.25 + p["phase"]) * ec
    x += 1.6 * math.sin(t * 0.9 + p["phase"] * 1.3) * ec
    return x, y, s, e

def title_block(cv, T, keys, ys, t, t_in, t_out, stagger=0.16):
    for i, (k, y) in enumerate(zip(keys, ys)):
        a_in = out_cubic((t - t_in - i * stagger) / 0.7); a_out = 1 - sstep(t_out, t_out + 0.45, t)
        a = a_in * a_out
        if a <= 0: continue
        L = T[k]; off = (1 - a_in) * 46 - (1 - a_out) * 26
        comp(cv, L, int(540 - L.shape[1] / 2), int(y - L.shape[0] / 2 + off), a)

def render(t, A):
    cv = A["bg"].copy(); T = A["T"]
    glow = np.zeros((H, W, 3), np.float32)
    z = cam_zoom(t); PAN[0] = key(PAN_KEYS, t)
    # ---- grid with soft parallax
    gz = 1 + (z - 1) * 0.45
    M = np.float32([[gz, 0, 540 - 700 * gz + 3 * math.sin(t * .2)], [0, gz, 960 - 1200 * gz - t * 2.0]])
    cv += cv2.warpAffine(A["grid"], M, (W, H), flags=cv2.INTER_LINEAR)
    # ---- drifting dust
    for dx, dy, sp, sz, ph in A["dust"]:
        x = (dx * W + 18 * math.sin(t * .4 + ph * 9)) % W; y = (dy * H - t * (10 + 26 * sp)) % H
        cv2.circle(glow, (int(x), int(y)), 1 + int(sz * 2), tuple(float(v) for v in BLUE * (0.10 + 0.25 * ph)), -1, cv2.LINE_AA)

    states = [(p,) + part_xform(p, t) for p in A["parts"]]
    E = float(np.mean([clamp(s[4]) for s in states]))
    hero_a = 1 - sstep(0.03, 0.26, E)

    # ---- floor ring under the machine
    fcx, fcy, _ = cam(540, 1010 + A["hero_h"] / 2 - 6, z)
    ring_a = 0.35 + 0.65 * hero_a
    ring = np.zeros((H, W, 3), np.float32)
    cv2.ellipse(ring, (int(fcx), int(fcy)), (int(440 * z), int(50 * z)), 0, 0, 360, tuple(float(v) for v in BLUE * 0.30 * ring_a), 2, cv2.LINE_AA)
    cv2.ellipse(ring, (int(fcx), int(fcy)), (int(560 * z), int(70 * z)), 0, t * 20, t * 20 + 110, tuple(float(v) for v in BLUE * 0.22 * ring_a), 2, cv2.LINE_AA)
    cv2.ellipse(ring, (int(fcx), int(fcy)), (int(560 * z), int(70 * z)), 0, t * 20 + 180, t * 20 + 290, tuple(float(v) for v in BLUE * 0.22 * ring_a), 2, cv2.LINE_AA)

    cv += ring + cv2.GaussianBlur(ring, (0, 0), 6) * 1.5

    # ---- machine photo (hero)
    if hero_a > 0.003:
        intro = sstep(0.0, 0.9, t)
        hx, hy, hz = cam(*A["hero_c"], z)
        f = PS * hz / A["hero_s"]
        draw_sprite(cv, A["hero_halo"], hx, hy, f, hero_a * intro * 0.8, add=True)
        rh = A["refl"].shape[0] * f
        draw_sprite(cv, A["refl"], hx, hy + A["hero"].shape[0] * f / 2 + rh / 2 - 8 * f, f, hero_a * intro)
        gain = 1.0
        draw_sprite(cv, A["hero"], hx, hy, f, hero_a * intro, gain=gain)
        # metallic light sweep across the machine
        for t0 in (1.3, 22.4):
            u = (t - t0) / 1.5
            if 0 < u < 1:
                hw, hh = A["hero"].shape[1] * f, A["hero"].shape[0] * f
                x0, y0 = int(hx - hw / 2), int(hy - hh / 2)
                xs = np.arange(max(x0, 0), min(int(x0 + hw), W)); ys = np.arange(max(y0, 0), min(int(y0 + hh), H))
                c = x0 - 150 + (hw + 300) * io_cubic(u)
                band = np.exp(-(((xs[None, :] - (ys[:, None] - hy) * 0.45) - c) / 55.0) ** 2).astype(np.float32)
                am = cv2.resize(A["hero"][..., 3], (int(hw), int(hh)))[ys[0] - y0:ys[0] - y0 + len(ys), xs[0] - x0:xs[0] - x0 + len(xs)]
                band = band[:am.shape[0], :am.shape[1]] * am * 70 * hero_a
                cv[ys[0]:ys[0] + band.shape[0], xs[0]:xs[0] + band.shape[1]] += band[..., None] * np.array([1.0, .93, .85], np.float32)

    # ---- component layers
    hl = {}
    for p in A["parts"]:
        if "li" in p:
            tl = T_LABELS + p["li"] * LABEL_STEP
            hl[p["id"]] = sstep(tl - .05, tl + .15, t) * (1 - sstep(tl + .25, tl + 1.1, t))
    dt = 1 / FPS
    for p, x, y, s, e in states:
        a = sstep(0.0, 0.14, e)
        if a <= 0.003: continue
        sx, sy, sz = cam(x, y, z, p["depth"])
        px, py, ps_, _ = part_xform(p, t - dt)
        qx, qy, _ = cam(px, py, cam_zoom(t - dt), p["depth"])
        speed = math.hypot(sx - qx, sy - qy)
        hxw, hyw, _ = cam(*p["home_w"], z, p["depth"])
        if speed > 1.2:                                              # energy trail + dashed explode axis
            k = clamp(speed / 22)
            n = max(int(math.hypot(sx - hxw, sy - hyw) / 14), 1)
            for i in range(0, n, 2):
                u0, u1 = i / n, min((i + 1) / n, 1)
                cv2.line(glow, (int(hxw + (sx - hxw) * u0), int(hyw + (sy - hyw) * u0)), (int(hxw + (sx - hxw) * u1), int(hyw + (sy - hyw) * u1)),
                         tuple(float(v) for v in BLUE * 0.28 * k), 1, cv2.LINE_AA)
            tx, ty = sx - (sx - qx) * 7, sy - (sy - qy) * 7
            cv2.line(glow, (int(tx), int(ty)), (int(sx), int(sy)), tuple(float(v) for v in BLUE * 0.75 * k), 3, cv2.LINE_AA)
        src = p["sp"] if s * sz > 1.0 else p["sp_half"]
        fbase = s * sz * (0.5 if src is p["sp"] else 1.0)
        if speed > 3:                                                # motion-blur ghosts
            for gi, ga in ((3, .10), (2, .17), (1, .27)):
                gx, gy, gs, ge = part_xform(p, t - dt * gi * 0.8)
                cx_, cy_, cz_ = cam(gx, gy, z, p["depth"])
                draw_sprite(cv, p["sp_half"], cx_, cy_, gs * cz_, a * ga)
        h = hl.get(p["id"], 0) + 0.5 * sstep(0, .3, speed / 20)
        if h > 0.01:
            draw_sprite(cv, p["halo"], sx, sy, s * sz, min(h, 1) * 0.75, add=True)
        draw_sprite(cv, src, sx, sy, fbase, a, gain=1 + 0.22 * hl.get(p["id"], 0))
        # lock pulse when a part seats during reassembly
        if t > T_REASM:
            tend = T_REASM + ((1.8 - p["delay"]) * 0.9 + REASM_DUR if p["id"] != "frame" else 3.9)
            u = (t - tend + 0.25) / 0.6
            if 0 < u < 1:
                c = tuple(float(v) for v in CYAN * (1 - u))
                cv2.circle(glow, (int(hxw), int(hyw)), int(8 + 46 * out_cubic(u)), c, 2, cv2.LINE_AA)
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.line(glow, (int(hxw + ddx * 10), int(hyw + ddy * 10)), (int(hxw + ddx * 22), int(hyw + ddy * 22)), c, 2, cv2.LINE_AA)
    for p, x, y, s, e in states:   # same pulses after the part has merged into the machine
        if t > T_REASM and sstep(0.0, 0.14, e) <= 0.003:
            tend = T_REASM + ((1.8 - p["delay"]) * 0.9 + REASM_DUR if p["id"] != "frame" else 3.9)
            u = (t - tend + 0.25) / 0.6
            if 0 < u < 1:
                hxw, hyw, _ = cam(*p["home_w"], z, p["depth"])
                cv2.circle(glow, (int(hxw), int(hyw)), int(8 + 46 * out_cubic(u)), tuple(float(v) for v in CYAN * (1 - u)), 2, cv2.LINE_AA)

    # ---- alignment brackets around the machine (scene 1, end of 4, scene 5)
    br = max(sstep(0.5, 1.4, t) * (1 - sstep(3.6, 4.2, t)), sstep(19.6, 20.8, t) * (1 - sstep(24.2, 24.9, t)) * 1.0)
    if br > 0.01:
        spread = 1 + 0.25 * (1 - sstep(19.6, 21.2, t)) if t > 10 else 1 + 0.12 * (1 - out_cubic((t - .5) / 1.0))
        bx0, by0, _ = cam(540 - 490 * spread, 1010 - 225 * spread, z); bx1, by1, _ = cam(540 + 490 * spread, 1010 + 235 * spread, z)
        c = tuple(float(v) for v in BLUE * br); L = 46
        for (x, y, dx, dy) in ((bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1)):
            cv2.line(glow, (int(x), int(y)), (int(x + dx * L), int(y)), c, 2, cv2.LINE_AA)
            cv2.line(glow, (int(x), int(y)), (int(x), int(y + dy * L)), c, 2, cv2.LINE_AA)
        ok = sstep(20.9, 21.3, t) * (1 - sstep(21.9, 22.3, t))
        if ok > 0: comp(cv, T["ok"], int(540 - T["ok"].shape[1] / 2), int(by1 + 6), ok)

    # ---- scan sweep at the moment of disassembly / final lock
    for t0, dur in ((3.75, 0.7), (20.35, 0.7)):
        u = (t - t0) / dur
        if 0 < u < 1:
            yl = int(cam(0, 780 + 470 * u, z)[1]); k = math.sin(math.pi * u)
            cv2.line(glow, (40, yl), (W - 40, yl), tuple(float(v) for v in CYAN * k), 3, cv2.LINE_AA)
            cv[max(yl - 60, 0):yl] += (np.linspace(0, 1, min(60, yl))[:, None, None] ** 2 * BLUE * 0.16 * k) if yl >= 60 else 0

    # ---- callouts
    pills = []
    for p, x, y, s, e in states:
        if "li" not in p: continue
        tl = T_LABELS + p["li"] * LABEL_STEP
        out = 1 - sstep(T_LABELS_OUT + p["li"] * 0.02, T_LABELS_OUT + 0.35 + p["li"] * 0.02, t)
        u = (t - tl) / 0.55
        if u <= 0 or out <= 0: continue
        sx, sy, sz = cam(x, y, z, p["depth"])
        tx, ty = sx + p["target"][0] * sz, sy + p["target"][1] * sz
        lx, ly, _ = cam(p["label_pos"][0], p["label_pos"][1], z)
        pill = p["pill"]; pw, ph = pill.shape[1], pill.shape[0]
        lx = clamp(lx, pw / 2 + 4, W - pw / 2 - 4)
        d = out_cubic(u / 0.3)                                        # dot
        col = tuple(float(v) for v in CYAN * out)
        cv2.circle(glow, (int(tx), int(ty)), int(2 + 3 * d), col, -1, cv2.LINE_AA)
        rp = (t - tl) / 0.7
        if 0 < rp < 1: cv2.circle(glow, (int(tx), int(ty)), int(6 + 26 * out_cubic(rp)), tuple(float(v) for v in BLUE * (1 - rp) * out), 2, cv2.LINE_AA)
        cv2.circle(glow, (int(tx), int(ty)), 9, tuple(float(v) for v in BLUE * 0.8 * d * out), 1, cv2.LINE_AA)
        l = out_cubic((u - 0.2) / 0.45)                               # leader line (elbow)
        if l > 0:
            ey_ = ly - ph / 2 + 5
            pts = [(tx, ty), (tx, ty + (ey_ - ty) * 0.45), (lx, ty + (ey_ - ty) * 0.45), (lx, ey_)]
            seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]; tot = sum(seg) * l
            for (a, b), sl in zip(zip(pts, pts[1:]), seg):
                if tot <= 0 or sl == 0: 
                    continue
                k = min(tot / sl, 1); tot -= sl
                cv2.line(glow, (int(a[0]), int(a[1])), (int(a[0] + (b[0] - a[0]) * k), int(a[1] + (b[1] - a[1]) * k)), tuple(float(v) for v in BLUE * 0.95 * out), 2, cv2.LINE_AA)
        w = out_cubic((u - 0.45) / 0.55)                              # pill wipes open from its centre
        if w > 0: pills.append((pill, lx, ly, w, out))

    # ---- glow pass
    small = cv2.resize(glow, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (W, H), interpolation=cv2.INTER_LINEAR)
    cv += glow + bloom * 1.7

    for pill, lx, ly, w, out in pills:
        pw, ph = pill.shape[1], pill.shape[0]
        half = max(int(pw / 2 * w), 1); c0 = pw // 2 - half
        comp(cv, pill[:, c0:c0 + 2 * half], int(lx - half), int(ly - ph / 2), out * min(w * 1.6, 1))

    # ---- titles
    title_block(cv, T, ["t1", "t2", "t3"], [545, 652, 762], t, 0.45, 3.45)
    a = out_cubic((t - 0.9) / 0.8) * (1 - sstep(3.45, 3.9, t))
    if a > 0: cv[709:712, int(540 - 210 * a):int(540 + 210 * a)] += BLUE * a
    title_block(cv, T, ["c1", "c2", "c3"], [545, 652, 762], t, T_FINAL + 0.15, 99)
    a = out_cubic((t - T_FINAL - 0.6) / 0.8)
    if a > 0: cv[709:712, int(540 - 210 * a):int(540 + 210 * a)] += BLUE * a
    # header (scenes 2-4)
    ha = sstep(4.3, 5.0, t) * (1 - sstep(21.2, 21.8, t))
    if ha > 0:
        L = T["h"]; comp(cv, L, int(540 - L.shape[1] / 2), int(128 - L.shape[0] / 2), ha)
        cv[171:173, int(540 - 300 * ha):int(540 + 300 * ha)] += BLUE * ha * 0.9
        for k, a0, a1 in (("h2", 4.5, 9.9), ("h3", 10.1, 15.9), ("h4", 16.1, 21.3)):
            a = sstep(a0, a0 + .4, t) * (1 - sstep(a1 - .3, a1, t)) * ha
            if a > 0:
                L = T[k]; comp(cv, L, int(540 - L.shape[1] / 2), int(206 - L.shape[0] / 2 + (1 - sstep(a0, a0 + .4, t)) * 14), a)
    # footer: process flow + progress bar
    fa = sstep(0.8, 1.6, t) * (1 - 0.0)
    step = 0 if t < 4 else 1 if t < 10 else 2 if t < 16 else 3 if t < 22 else 4
    xs = [118, 310, 512, 734, 950]
    for i, x in enumerate(xs):
        L = T[f"s{i}on" if i == step else f"s{i}off"]
        comp(cv, L, int(x - L.shape[1] / 2), int(1848 - L.shape[0] / 2), fa)
    cv[1818:1820, 60:W - 60] += BLUE * 0.18 * fa
    cv[1817:1821, 60:int(60 + (W - 120) * t / DUR)] += BLUE * 0.9 * fa
    for (x, y, dx, dy) in ((36, 36, 1, 1), (W - 36, 36, -1, 1), (36, H - 36, 1, -1), (W - 36, H - 36, -1, -1)):   # corner ticks
        cv2.line(cv, (x, y), (x + dx * 34, y), tuple(float(v) for v in BLUE * 0.6), 2, cv2.LINE_AA)
        cv2.line(cv, (x, y), (x, y + dy * 34), tuple(float(v) for v in BLUE * 0.6), 2, cv2.LINE_AA)

    # ---- finishing: flashes, vignette, grain, fades
    for t0 in (4.1, 20.75):
        fl = math.exp(-((t - t0) / 0.16) ** 2) * 22
        if fl > .3: cv += BLUE / 255 * fl + fl * 0.35
    cv *= A["vig"]
    cv += cv2.resize(A["grain"][int(t * FPS) % 8], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    cv *= sstep(0, 0.5, t) * (1 - sstep(24.55, 25.0, t))
    return np.clip(cv, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    A = load_assets()
    if "--still" in sys.argv:
        for v in sys.argv[sys.argv.index("--still") + 1:]:
            cv2.imwrite(f"{R}/build/still_{float(v):05.2f}.png", render(float(v), A))
        sys.exit()
    os.makedirs(f"{R}/out", exist_ok=True)
    out = f"{R}/out/cigarette_machine_exploded_view.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        ff.stdin.write(render(i / FPS, A).tobytes())
        if i % 60 == 0: print(f"{i}/{n}", flush=True)
    ff.stdin.close(); ff.wait(); print("wrote", out)
