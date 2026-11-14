"""MK-8 exploded-view reel (1080x1920, 30 fps) built from the reference photos.

Usage: python3 render.py [--still SECONDS] [--frames DIR]
Every moving part is a real cut-out of the reference photo, so geometry and
identity of the machine are preserved exactly.
"""
import math, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
DUR = 41.0
CYAN = (90, 200, 255)
WHITE = (255, 255, 255)

F_BOLD = '/usr/share/fonts/opentype/inter/Inter-Bold.otf'
F_SEMI = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
F_REG = '/usr/share/fonts/opentype/inter/Inter-Regular.otf'
F_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
for p in (F_SEMI, F_REG):
    if not os.path.exists(p):
        F_SEMI = F_REG = F_BOLD
_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

def load(i):
    return Image.open(os.path.join(HERE, 'ref', f'{i}.jpg')).convert('RGB')

IMG = {i: load(i) for i in range(1, 6)}

# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))
def ease(x):
    x = clamp(x)
    return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3
def ease_back(x):
    x = clamp(x); c1 = 1.4; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def seg(t, a, b):
    return clamp((t - a) / (b - a))
def lerp(a, b, k):
    return a + (b - a) * k

# ---------------------------------------------------------------- backgrounds
def make_bg(img):
    """Blurred, darkened cover version of the photo (studio backdrop)."""
    s = max(W / img.width, H / img.height) * 1.08
    b = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
    l, t = (b.width - W) // 2, (b.height - H) // 2
    b = b.crop((l, t, l + W, t + H)).filter(ImageFilter.GaussianBlur(40))
    a = np.asarray(b).astype(np.float32) * 0.28
    a = a * np.array([0.85, 0.95, 1.15])  # cool tint
    yy, xx = np.mgrid[0:H, 0:W]
    v = 1 - 0.55 * (((xx - W / 2) / (W * .75)) ** 2 + ((yy - H / 2) / (H * .6)) ** 2)
    a = a * np.clip(v, 0.25, 1)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def studio_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 1 - np.clip(np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * .45) / H) ** 2) * 1.5, 0, 1)
    base = np.stack([10 + 22 * g, 14 + 30 * g, 22 + 46 * g], -1)
    # faint engineering grid
    grid = ((xx % 60 < 1) | (yy % 60 < 1)).astype(np.float32) * 10 * g
    base += grid[..., None] * np.array([0.4, 0.8, 1.0])
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

BG = {i: make_bg(IMG[i]) for i in IMG}
STUDIO = studio_bg()

def feather_mask(size, r):
    m = Image.new('L', size, 0)
    ImageDraw.Draw(m).rectangle((r, r, size[0] - r, size[1] - r), fill=255)
    return m.filter(ImageFilter.GaussianBlur(r * 0.6))

PLATE = {i: IMG[i].copy() for i in IMG}
for i in PLATE:
    PLATE[i].putalpha(feather_mask(IMG[i].size, 40))

def view(img_rgba, cx, cy, s):
    """Place source point (cx,cy) at frame centre with scale s (affine, subpixel)."""
    a = 1 / s
    return img_rgba.transform((W, H), Image.AFFINE,
                              (a, 0, cx - W / 2 * a, 0, a, cy - H / 2 * a),
                              resample=Image.BICUBIC)

def bg_shift(bg, dx):
    return ImageChops.offset(bg, int(dx), 0)

# ---------------------------------------------------------------- HUD bits
def hud_frame(d, alpha, caption):
    a = int(200 * alpha)
    if a <= 0:
        return
    c = CYAN + (a,)
    L = 60; m = 40
    for (x, y, sx, sy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
        d.line([(x, y), (x + sx * L, y)], fill=c, width=3)
        d.line([(x, y), (x, y + sy * L)], fill=c, width=3)
    d.rectangle((m, m + 70, m + 520, m + 140), fill=(6, 12, 22, int(150 * alpha)))
    d.text((m + 14, m + 78), 'MK-8  //  ENGINEERING VIEW', font=font(F_MONO, 22), fill=WHITE + (int(200 * alpha),))
    if caption:
        d.text((m + 14, m + 108), caption, font=font(F_MONO, 20), fill=CYAN + (a,))

def label(d, anchor, box_xy, text, k, align='left', small=False):
    """Leader line grows from anchor, then pill label fades in. k in [0,1]."""
    if k <= 0:
        return
    ax, ay = anchor; bx, by = box_xy
    kl = ease_out(k / 0.5)
    kt = ease_out((k - 0.35) / 0.65)
    # elbow: anchor -> (bx, by) horizontal into box
    ex, ey = lerp(ax, bx, kl), lerp(ay, by, kl)
    d.line([(ax, ay), (ex, ey)], fill=WHITE + (230,), width=2)
    r = 7
    d.ellipse((ax - r, ay - r, ax + r, ay + r), outline=CYAN + (255,), width=2)
    d.ellipse((ax - 3, ay - 3, ax + 3, ay + 3), fill=WHITE + (255,))
    if kt <= 0:
        return
    f = font(F_SEMI, 25 if not small else 23)
    n = max(1, int(round(len(text) * min(1, kt * 1.3))))
    shown = text[:n]
    tw = f.getlength(text); th = 30
    pad = 12
    bw = tw + pad * 2 + 6
    x0 = bx if align == 'left' else bx - bw
    y0 = by - th / 2 - 6
    A = int(255 * kt)
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + th + 12), radius=6,
                        fill=(8, 18, 32, int(205 * kt)), outline=WHITE + (int(110 * kt),), width=1)
    d.rectangle((x0, y0 + 4, x0 + 4, y0 + th + 8), fill=CYAN + (A,))
    d.text((x0 + pad + 6, y0 + 7), shown, font=f, fill=WHITE + (A,))

def title_card(d, alpha, y, big, small):
    if alpha <= 0:
        return
    A = int(255 * alpha)
    fb = font(F_BOLD, 120); fs = font(F_SEMI, 40); fm = font(F_MONO, 24)
    w = fb.getlength(big)
    d.text(((W - w) / 2, y), big, font=fb, fill=WHITE + (A,))
    ln = 360 * ease_out(alpha)
    d.line([(W / 2 - ln / 2, y + 150), (W / 2 + ln / 2, y + 150)], fill=CYAN + (A,), width=3)
    for i, s in enumerate(small.split('\n')):
        w2 = fs.getlength(s)
        d.text(((W - w2) / 2, y + 175 + i * 52), s, font=fs, fill=WHITE + (int(A * .92),))

# ---------------------------------------------------------------- exploded-view model (photo 2)
SRC = IMG[2]
EXP_C = (980, 560)        # source point at frame centre
Z0, Z1 = 0.68, 0.56       # camera scale assembled / exploded

# name, source rect, exploded target centre (canvas, at Z1), label pos offset from part centre, align, delay
PARTS = [
    ('Fluid Reservoir',             (708, 30, 838, 192),   (205, 300),  (70, -40),  'left',  0.00),
    ('Drive Motor',                 (1045, 122, 1245, 360), (545, 330), (-45, -112), 'left',  0.10),
    ('Material Feeding Hopper',     (1238, 82, 1642, 352), (850, 470),  (-150, -120), 'left', 0.05),
    ('Electrical Control Panel',    (630, 238, 990, 392),  (300, 560),  (-230, 75), 'left',   0.15),
    ('Air Regulator & Valves',      (738, 398, 862, 502),  (115, 760),  (-80, -75), 'left',   0.30),
    ('Conveyor & Transfer Assembly', (985, 338, 1692, 522), (690, 670), (-330, 40), 'left',   0.20),
    ('Side Housing',                (325, 402, 500, 692),  (110, 1430), (-70, 112), 'left',  0.35),
    ('Rotary Drum Assembly',        (500, 518, 738, 722),  (350, 1420), (-115, -95), 'left',  0.30),
    ('Roller & Gear Assembly',      (738, 556, 1128, 802), (700, 1440), (-110, -105), 'left', 0.25),
    ('Protective Guard',            (1262, 478, 1628, 642), (945, 1240), (100, -60), 'right', 0.40),
    ('Collection Tray',             (308, 752, 668, 918),  (210, 1660), (-120, 70), 'left',   0.45),
    ('Adjustment Bar & Dials',      (1272, 712, 1618, 802), (880, 1620), (-110, -42), 'left', 0.50),
    ('Perforated Tray',             (1270, 818, 1628, 908), (880, 1745), (-90, 50), 'left',   0.55),
    ('Sensor / Switch Bracket',     (1010, 818, 1142, 932), (580, 1680), (-120, 75), 'left',  0.60),
    ('Handwheel',                   (632, 876, 722, 958),  (470, 1570), (40, 0), 'left',      0.62),
    ('Ventilated Housing',          (222, 818, 382, 972),  (110, 1820), (60, 0), 'left',      0.65),
]

def to_canvas(x, y, z, c=EXP_C):
    return W / 2 + (x - c[0]) * z, H / 2 + (y - c[1]) * z

def build_parts():
    out = []
    for name, r, tgt, lo, al, dl in PARTS:
        sp = SRC.crop(r).convert('RGBA')
        m = Image.new('L', sp.size, 255)
        m = m.filter(ImageFilter.GaussianBlur(0))
        sp.putalpha(m)
        sh = Image.new('L', (sp.width + 80, sp.height + 80), 0)
        ImageDraw.Draw(sh).rectangle((40, 40, 40 + sp.width, 40 + sp.height), fill=255)
        sh = sh.filter(ImageFilter.GaussianBlur(18))
        out.append(dict(name=name, rect=r, tgt=tgt, lo=lo, al=al, dl=dl, sp=sp, sh=sh))
    return out

PART = build_parts()

# frame layer: holes become dark recesses with a cyan socket outline
def build_frame_layers():
    base = SRC.copy()
    rec = np.asarray(SRC).astype(np.float32)
    rec = (rec * 0.18 + np.array([4, 10, 18])).clip(0, 255).astype(np.uint8)
    rec = Image.fromarray(rec)
    hole = base.copy()
    mask = Image.new('L', SRC.size, 0)
    md = ImageDraw.Draw(mask)
    for p in PART:
        md.rectangle(p['rect'], fill=255)
    hole.paste(rec, (0, 0), mask)
    hd = ImageDraw.Draw(hole)
    for p in PART:
        x0, y0, x1, y1 = p['rect']
        hd.rectangle((x0, y0, x1, y1), outline=CYAN, width=3)
    fm = feather_mask(SRC.size, 40)
    a = base.convert('RGBA'); a.putalpha(fm)
    b = hole.convert('RGBA'); b.putalpha(fm)
    return a, b

FRAME_FULL, FRAME_HOLES = build_frame_layers()

# timing (seconds)
T_EXP0, T_EXP1 = 8.0, 12.5      # explode
T_LAB0 = 12.4                  # labels start
T_CLOSE0 = 18.6                # close-ups start
T_BACK = 30.5                  # back to exploded view
T_ASM0, T_ASM1 = 31.2, 35.6    # reassembly
T_HERO = 36.0
PART_DUR = 2.6

def explode_k(p, t):
    """0 = assembled, 1 = exploded."""
    if t < T_ASM0 - 0.01:
        st = T_EXP0 + p['dl'] * (T_EXP1 - T_EXP0 - PART_DUR) / 0.65
        return ease(seg(t, st, st + PART_DUR))
    # reassembly: reverse order of delays
    st = T_ASM0 + (0.65 - p['dl']) * (T_ASM1 - T_ASM0 - PART_DUR) / 0.65
    return 1 - ease(seg(t, st, st + PART_DUR))

def asm_end(p):
    return T_ASM0 + (0.65 - p['dl']) * (T_ASM1 - T_ASM0 - PART_DUR) / 0.65 + PART_DUR

def cam_z(t):
    if t < T_ASM0:
        return lerp(Z0, Z1, ease(seg(t, T_EXP0 - 0.3, T_EXP1)))
    return lerp(Z1, Z0, ease(seg(t, T_ASM0, T_ASM1 + 0.3)))

def part_state(p, t, z, k):
    x0, y0, x1, y1 = p['rect']
    rcx, rcy = to_canvas((x0 + x1) / 2, (y0 + y1) / 2, z)
    # exploded target defined at Z1; scale offsets with camera
    tx, ty = W / 2 + (p['tgt'][0] - W / 2) * z / Z1, H / 2 + (p['tgt'][1] - H / 2) * z / Z1
    # arc path: lift toward camera mid-flight
    cx = lerp(rcx, tx, k); cy = lerp(rcy, ty, k)
    bob = 0.0
    if k > 0.99:
        ph = hash(p['name']) % 100 / 100 * 6.28
        bob = 4 * math.sin(t * 1.6 + ph)
    sc = z * (1 + 0.10 * math.sin(math.pi * k) + 0.06 * k)
    return cx, cy + bob, sc

def paste_center(canvas, im, cx, cy):
    canvas.alpha_composite(im, (int(round(cx - im.width / 2)), int(round(cy - im.height / 2))))

def render_exploded(t, show_labels, caption):
    z = cam_z(t)
    frame = STUDIO.copy().convert('RGBA')
    ks = [explode_k(p, t) for p in PART]
    kmax = max(ks)
    # frame plate (blend to holes version as parts leave)
    fl = view(FRAME_FULL, EXP_C[0], EXP_C[1], z)
    if kmax > 0:
        fh = view(FRAME_HOLES, EXP_C[0], EXP_C[1], z)
        fl = Image.blend(fl, fh, clamp(kmax * 3))
    frame.alpha_composite(fl)
    fx = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fx)
    # light trails during reassembly + explode streaks
    for p, k in zip(PART, ks):
        if 0.02 < k < 0.98:
            pts = []
            for j in range(10):
                tt = t - j * 0.035
                kk = explode_k(p, tt)
                cx, cy, _ = part_state(p, tt, cam_z(tt), kk)
                pts.append((cx, cy))
            for j in range(len(pts) - 1):
                a = int(220 * (1 - j / 9))
                fd.line([pts[j], pts[j + 1]], fill=CYAN + (a,), width=6 - j // 2)
    # parts, back to front: draw those with larger offsets last
    order = sorted(range(len(PART)), key=lambda i: ks[i])
    for i in order:
        p, k = PART[i], ks[i]
        cx, cy, sc = part_state(p, t, z, k)
        sw, shh = max(1, int(p['sp'].width * sc)), max(1, int(p['sp'].height * sc))
        if k > 0.01:
            sh = p['sh'].resize((max(1, int(p['sh'].width * sc)), max(1, int(p['sh'].height * sc))))
            sha = Image.new('RGBA', sh.size, (0, 0, 0, 0))
            sha.putalpha(sh.point(lambda v, kk=k: int(v * 0.65 * min(1, kk * 2))))
            paste_center(frame, sha, cx + 14 * k, cy + 26 * k)
        sp = p['sp'].resize((sw, shh), Image.BICUBIC)
        paste_center(frame, sp, cx, cy)
        if k > 0.02:
            x0, y0 = cx - sw / 2, cy - shh / 2
            fd.rectangle((x0 - 3, y0 - 3, x0 + sw + 3, y0 + shh + 3), outline=CYAN + (int(140 * min(1, k * 2)),), width=2)
            L = 14
            for (xx, yy, sx, sy) in ((x0 - 8, y0 - 8, 1, 1), (x0 + sw + 8, y0 - 8, -1, 1),
                                     (x0 - 8, y0 + shh + 8, 1, -1), (x0 + sw + 8, y0 + shh + 8, -1, -1)):
                fd.line([(xx, yy), (xx + sx * L, yy)], fill=WHITE + (int(220 * min(1, k * 2)),), width=2)
                fd.line([(xx, yy), (xx, yy + sy * L)], fill=WHITE + (int(220 * min(1, k * 2)),), width=2)
        # lock flash after reassembly
        if t >= T_ASM0:
            dt = t - asm_end(p)
            if 0 <= dt < 0.5:
                q = dt / 0.5
                rr = 20 + 90 * ease_out(q)
                a = int(255 * (1 - q))
                fd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=CYAN + (a,), width=4)
                x0, y0 = cx - sw / 2, cy - shh / 2
                fd.rectangle((x0, y0, x0 + sw, y0 + shh), outline=WHITE + (a,), width=3)
    glow = fx.filter(ImageFilter.GaussianBlur(6))
    frame.alpha_composite(glow)
    frame.alpha_composite(fx)
    # labels
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if show_labels > 0:
        n = len(PART) + 1
        for idx, p in enumerate(PART):
            k = ks[idx]
            if k < 0.98:
                continue
            cx, cy, sc = part_state(p, t, z, k)
            lk = seg(t, T_LAB0 + idx * 0.27, T_LAB0 + idx * 0.27 + 0.6) * show_labels
            label(d, (cx, cy), (cx + p['lo'][0], cy + p['lo'][1]), p['name'], lk, p['al'])
        # main frame
        fxp, fyp = to_canvas(1150, 960, z)
        lk = seg(t, T_LAB0 + n * 0.27, T_LAB0 + n * 0.27 + 0.6) * show_labels
        label(d, (fxp, fyp), (fxp - 300, fyp + 10), 'Main Machine Frame', lk, 'left')
    return frame, ov, d

# ---------------------------------------------------------------- close-ups
CLOSE = [
    # (image, t0, t1, path (cx,cy,s) start, end, labels[(src pt, text, side)])
    (4, 18.6, 23.8, (470, 560, 2.05), (1480, 600, 2.05), [
        ((235, 160), 'Pneumatic Solenoid Valves', 'right'),
        ((565, 650), 'Grinding Wheel', 'right'),
        ((460, 935), 'Pulley', 'right'),
        ((715, 160), 'Rotary Drum', 'right'),
        ((925, 600), 'Rotary Drum', 'left'),
        ((960, 940), 'Gears', 'right'),
        ((1290, 330), 'Timing Pulley', 'left'),
        ((1350, 720), 'Toothed Transfer Drum', 'left'),
        ((1480, 340), 'Limit Switch', 'left'),
        ((1700, 330), 'Pneumatic Hoses', 'left'),
        ((1810, 840), 'Drum Assembly', 'left'),
    ]),
    (3, 23.8, 28.1, (295, 580, 1.78), (1290, 600, 1.85), [
        ((280, 370), 'Main Power Isolator', 'right'),
        ((285, 505), 'Drum Temperature Display', 'right'),
        ((290, 700), 'Gum / Heater Controls', 'right'),
        ((290, 900), 'Knife / Filter Selectors', 'right'),
        ((790, 650), 'Transparent Safety Guard', 'right'),
        ((1165, 400), 'Gear Train', 'left'),
        ((1135, 850), 'Hand Lever', 'left'),
        ((1530, 800), 'Reel Spindle', 'left'),
    ]),
    (5, 28.1, 30.6, (1000, 470, 1.62), (1000, 560, 1.78), [
        ((990, 160), 'Protective Cover', 'right'),
        ((990, 545), 'Adjustment Bar & Dials', 'right'),
        ((990, 710), 'Perforated Tray', 'right'),
    ]),
]

def render_close(t, ci):
    im, t0, t1, a, b, labs = CLOSE[ci]
    k = ease(seg(t, t0, t1))
    cx, cy, s = (lerp(a[i], b[i], k) for i in range(3))
    frame = BG[im].copy().convert('RGBA')
    v = view(PLATE[im], cx, cy, s)
    frame.alpha_composite(v)
    # depth of field: blur top & bottom bands
    rgb = frame.convert('RGB')
    bl = rgb.filter(ImageFilter.GaussianBlur(9))
    yy = np.linspace(0, 1, H)
    m = np.clip(np.maximum((0.22 - yy) / 0.18, (yy - 0.78) / 0.18), 0, 1)
    mask = Image.fromarray((np.repeat(m[:, None], W, 1) * 255).astype(np.uint8))
    rgb.paste(bl, (0, 0), mask)
    frame = rgb.convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for (sx, sy), text, side in labs:
        X, Y = W / 2 + (sx - cx) * s, H / 2 + (sy - cy) * s
        vis = clamp(1 - max(abs(X - W / 2) - 400, 0) / 90) * clamp(1 - max(abs(Y - H / 2) - 700, 0) / 100)
        if vis <= 0:
            continue
        lk = clamp(vis * 1.0)
        bx = X + 60 if side == 'right' else X - 60
        bx = clamp(bx, 50, W - 50)
        label(d, (X, Y), (bx, Y - 50), text, lk, 'left' if side == 'right' else 'right')
    # fade between close-ups
    fade = min(seg(t, t0, t0 + 0.35), 1 - seg(t, t1 - 0.3, t1))
    return frame, ov, d, fade

# ---------------------------------------------------------------- full-line shots (photo 1)
def render_line(t):
    if t < 7.0:
        # dolly in + slow drift (pseudo orbit via parallax)
        k = ease(seg(t, 0.2, 5.0))
        cx = lerp(1000, 1320, k); cy = lerp(560, 650, k); s = lerp(0.54, 1.80, k)
        k2 = ease(seg(t, 5.0, 7.0))
        cx += lerp(0, -90, k2); s *= lerp(1, 1.04, k2)
        par = -60 * ease(seg(t, 0, 7))
    else:
        # hero pull-back
        k = ease(seg(t, T_HERO, DUR - 1.0))
        cx = lerp(1240, 960, k); cy = lerp(660, 620, k); s = lerp(1.95, 0.56, k)
        par = 60 * k
    frame = bg_shift(BG[1], par).convert('RGBA')
    frame.alpha_composite(view(PLATE[1], cx, cy, s))
    return frame

# ---------------------------------------------------------------- frame compositor
def render(fi):
    t = fi / FPS
    caption = ''
    if t < 7.0:
        frame = render_line(t)
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        a = seg(t, 0.4, 1.3) * (1 - seg(t, 2.6, 3.4))
        title_card(d, a, 260, 'MK-8', 'Automatic Cigarette\nMaking Machine')
        hud_frame(d, seg(t, 4.5, 5.5), '')
        # transition to front view
        if t > 6.4:
            ex, eov, ed = render_exploded(T_EXP0 - 1.0, 0, '')
            q = ease(seg(t, 6.4, 7.0))
            frame = Image.blend(frame, ex, q)
        frame.alpha_composite(ov)
    elif t < T_CLOSE0 or (T_BACK <= t < T_HERO):
        if t < T_EXP0:
            caption = '> FRONT ELEVATION  //  ASSEMBLED'
        elif t < T_EXP1:
            caption = '> EXPLODED VIEW  //  SEPARATING ASSEMBLIES'
        elif t < T_CLOSE0:
            caption = '> COMPONENT IDENTIFICATION'
        elif t < T_ASM0:
            caption = '> PREPARING REASSEMBLY'
        else:
            caption = '> REASSEMBLY  //  PRECISION ALIGNMENT'
        sl = 0.0
        if T_LAB0 <= t < T_CLOSE0:
            sl = 1 - seg(t, T_CLOSE0 - 0.4, T_CLOSE0)
        frame, ov, d = render_exploded(t, sl, caption)
        hud_frame(d, 1.0, caption)
        if T_HERO - 0.5 < t < T_HERO:
            q = seg(t, T_HERO - 0.5, T_HERO)
            frame = Image.blend(frame, render_line(T_HERO), ease(q))
        if T_BACK <= t < T_BACK + 0.4:
            frame = Image.blend(Image.new('RGBA', (W, H), (0, 0, 0, 255)), frame, seg(t, T_BACK, T_BACK + 0.4))
        frame.alpha_composite(ov)
    elif t < T_BACK:
        ci = 0 if t < CLOSE[0][2] else (1 if t < CLOSE[1][2] else 2)
        frame, ov, d, fade = render_close(t, ci)
        hud_frame(d, 1.0, '> DETAIL INSPECTION  //  ' + ['ROTARY DRUMS & PULLEYS', 'CONTROLS & DRIVE', 'TRAY & GUARD'][ci])
        frame.alpha_composite(ov)
        frame = Image.blend(Image.new('RGBA', (W, H), (0, 0, 0, 255)), frame, fade)
    else:
        frame = render_line(t)
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        a = seg(t, 38.0, 39.0)
        title_card(d, a, 230, 'MK-8', 'Automatic Cigarette\nMaking Machine')
        if a > 0:
            fm = font(F_MONO, 24); s = 'PRECISION  ·  RELIABILITY  ·  PERFORMANCE'
            d.text(((W - fm.getlength(s)) / 2, 1620), s, font=fm, fill=CYAN + (int(255 * a),))
        hud_frame(d, 1 - seg(t, 36.5, 37.5), '')
        frame.alpha_composite(ov)
        if t > DUR - 0.6:
            frame = Image.blend(frame, Image.new('RGBA', (W, H), (0, 0, 0, 255)), seg(t, DUR - 0.6, DUR))
    # global vignette + light grain
    out = frame.convert('RGB')
    return out

VIG = None
def finish(img, fi):
    global VIG
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        VIG = np.clip(1 - 0.35 * (((xx - W / 2) / (W * .7)) ** 2 + ((yy - H / 2) / (H * .7)) ** 2), 0.55, 1)[..., None]
    a = np.asarray(img).astype(np.float32) * VIG
    rng = np.random.default_rng(fi)
    a += rng.normal(0, 3.0, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def work(fi):
    out_dir = os.environ['FRAMES']
    finish(render(fi), fi).save(os.path.join(out_dir, f'{fi:05d}.jpg'), quality=93)
    return fi

if __name__ == '__main__':
    if '--still' in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index('--still') + 1].split(',')]
        od = sys.argv[sys.argv.index('--still') + 2]
        for t in ts:
            finish(render(int(t * FPS)), 0).save(os.path.join(od, f'still_{t:05.1f}.jpg'), quality=90)
        sys.exit()
    os.makedirs(os.environ['FRAMES'], exist_ok=True)
    n = int(DUR * FPS)
    with Pool(4) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, range(n), chunksize=8)):
            if i % 120 == 0:
                print(i, '/', n, flush=True)
