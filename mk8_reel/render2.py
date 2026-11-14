"""MK-8 animated exploded-view reel v2 (1080x1920, 30 fps).

Animated version: wheels/drums/gears rotate inside the real photos, exploded parts are
true 3D cards (perspective, thickness, depth sort) under an orbiting camera, wireframe
scan, glitch cuts. Every visible part is pixels from the reference photos.

python3 render2.py                 -> frames into $FRAMES
python3 render2.py --still 5,12 D  -> stills into D
"""
import math, os, sys, json
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
DUR = 36.0
CYAN = (90, 200, 255)
WHITE = (255, 255, 255)
FOC = 1900.0

F_BOLD = '/usr/share/fonts/opentype/inter/Inter-Bold.otf'
F_SEMI = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
F_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
if not os.path.exists(F_SEMI):
    F_SEMI = F_BOLD
_fc = {}
def font(p, s):
    if (p, s) not in _fc:
        _fc[(p, s)] = ImageFont.truetype(p, s)
    return _fc[(p, s)]

IMG = {i: Image.open(os.path.join(HERE, 'ref', f'{i}.jpg')).convert('RGB') for i in range(1, 6)}
NP = {i: np.asarray(IMG[i]) for i in IMG}
WHEELS = {int(k): v for k, v in json.load(open(os.path.join(HERE, 'wheels.json'))).items()}

# ---------------------------------------------------------------- timeline (120 BPM, 2 s bars)
T_SCAN0, T_SCAN1 = 4.0, 6.0
T_EXP0, T_EXP1 = 6.0, 10.0
T_LAB0 = 9.8
T_CLOSE = [(16.0, 20.0, 4), (20.0, 24.0, 3), (24.0, 26.0, 5)]
T_BACK = 26.0
T_ASM0, T_ASM1 = 26.6, 31.0
T_HERO = 32.0
CUTS = [4.0, 16.0, 20.0, 24.0, 26.0, 32.0]
PART_DUR = 2.4

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(x):
    x = clamp(x); return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def ease_out(x):
    x = clamp(x); return 1 - (1 - x) ** 3
def seg(t, a, b): return clamp((t - a) / (b - a))
def lerp(a, b, k): return a + (b - a) * k

# ---------------------------------------------------------------- animated source photos
def anim_src(i, t):
    """Photo i with its real wheels rotating about their own centres."""
    if i not in WHEELS:
        return IMG[i]
    PADW = 220
    src = np.pad(NP[i], ((PADW, PADW), (PADW, PADW), (0, 0)), mode='edge')
    a = src.copy()
    for (x, y, r, sp) in WHEELS[i]:
        ang = sp * 70.0 * t
        p = r + 4
        x, y = x + PADW, y + PADW
        x0, y0, x1, y1 = x - p, y - p, x + p, y + p
        crop = src[y0:y1, x0:x1]
        M = cv2.getRotationMatrix2D((float(p), float(p)), ang, 1)
        rot = cv2.warpAffine(crop, M, (2 * p, 2 * p), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        yy, xx = np.ogrid[:2 * p, :2 * p]
        d = np.sqrt((xx - p) ** 2 + (yy - p) ** 2)
        m = np.clip(r - d + 0.5, 0, 1)[..., None]
        a[y0:y1, x0:x1] = (rot * m + a[y0:y1, x0:x1] * (1 - m)).astype(np.uint8)
    return Image.fromarray(a[PADW:-PADW, PADW:-PADW])

def feather(size, r):
    m = Image.new('L', size, 0)
    ImageDraw.Draw(m).rectangle((r, r, size[0] - r, size[1] - r), fill=255)
    return m.filter(ImageFilter.GaussianBlur(r * 0.6))
FEATHER = feather((2000, 1125), 40)

def rgba(img, mask=FEATHER):
    o = img.convert('RGBA'); o.putalpha(mask); return o

def make_bg(img):
    s = max(W / img.width, H / img.height) * 1.1
    b = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
    l, t = (b.width - W) // 2, (b.height - H) // 2
    b = b.crop((l, t, l + W, t + H)).filter(ImageFilter.GaussianBlur(40))
    a = np.asarray(b).astype(np.float32) * 0.26 * np.array([0.85, 0.95, 1.15])
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
BG = {i: make_bg(IMG[i]) for i in IMG}

def studio():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 1 - np.clip(np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * .45) / H) ** 2) * 1.5, 0, 1)
    base = np.stack([10 + 22 * g, 14 + 30 * g, 22 + 46 * g], -1)
    base += (((xx % 60 < 1) | (yy % 60 < 1)) * 10 * g)[..., None] * np.array([0.4, 0.8, 1.0])
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
STUDIO = studio()

EDGES = {}
for i in (1, 2):
    g = cv2.GaussianBlur(cv2.cvtColor(NP[i], cv2.COLOR_RGB2GRAY), (3, 3), 0)
    e = cv2.Canny(g, 60, 150)
    e = cv2.dilate(e, np.ones((2, 2), np.uint8))
    EDGES[i] = e.astype(np.float32) / 255

# ---------------------------------------------------------------- 3D camera
class Cam:
    """Source-plane coords -> screen with yaw orbit. (cx,cy) source point at centre, scale z."""
    def __init__(s, cx, cy, z, yaw):
        s.cx, s.cy, s.z, s.yaw = cx, cy, z, math.radians(yaw)
        s.c, s.s = math.cos(s.yaw), math.sin(s.yaw)
    def world(s, sx, sy):
        return (sx - s.cx) * s.z, (sy - s.cy) * s.z
    def proj(s, X, Y, Z=0.0):
        x = X * s.c + Z * s.s
        z = -X * s.s + Z * s.c
        k = FOC / (FOC + z)
        return W / 2 + x * k, H / 2 + Y * k, k

def coeffs(dst, src):
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()

def warp_plane(img, src_quad, dst_quad, out_size=(W, H), off=(0, 0)):
    d = [(x - off[0], y - off[1]) for x, y in dst_quad]
    return img.transform(out_size, Image.PERSPECTIVE, coeffs(d, src_quad), resample=Image.BICUBIC)

def plate(img_rgba, cam):
    w, h = img_rgba.size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [cam.proj(*cam.world(x, y))[:2] for x, y in src]
    return warp_plane(img_rgba, src, dst)

# ---------------------------------------------------------------- HUD
def hud(d, alpha, caption):
    a = int(200 * alpha)
    if a <= 0: return
    c = CYAN + (a,); L = 60; m = 40
    for (x, y, sx, sy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
        d.line([(x, y), (x + sx * L, y)], fill=c, width=3); d.line([(x, y), (x, y + sy * L)], fill=c, width=3)
    d.rectangle((m, m + 70, m + 540, m + 140), fill=(6, 12, 22, int(150 * alpha)))
    d.text((m + 14, m + 78), 'MK-8  //  ENGINEERING VIEW', font=font(F_MONO, 22), fill=WHITE + (int(200 * alpha),))
    if caption:
        d.text((m + 14, m + 108), caption, font=font(F_MONO, 20), fill=c)

def label(d, anchor, box, text, k, align='left'):
    if k <= 0: return
    ax, ay = anchor; bx, by = box
    kl = ease_out(k / 0.5); kt = ease_out((k - 0.35) / 0.65)
    d.line([(ax, ay), (lerp(ax, bx, kl), lerp(ay, by, kl))], fill=WHITE + (230,), width=2)
    d.ellipse((ax - 7, ay - 7, ax + 7, ay + 7), outline=CYAN + (255,), width=2)
    d.ellipse((ax - 3, ay - 3, ax + 3, ay + 3), fill=WHITE + (255,))
    if kt <= 0: return
    f = font(F_SEMI, 25)
    shown = text[:max(1, int(round(len(text) * min(1, kt * 1.3))))]
    bw = f.getlength(text) + 30; th = 30
    x0 = bx if align == 'left' else bx - bw
    x0 = clamp(x0, 20, W - 20 - bw)
    y0 = by - th / 2 - 6; A = int(255 * kt)
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + th + 12), radius=6, fill=(8, 18, 32, int(205 * kt)),
                        outline=WHITE + (int(110 * kt),), width=1)
    d.rectangle((x0, y0 + 4, x0 + 4, y0 + th + 8), fill=CYAN + (A,))
    d.text((x0 + 18, y0 + 7), shown, font=f, fill=WHITE + (A,))

def title(d, t, t0, t1, y):
    """Kinetic title: letters drop in, then hold, then fade."""
    a_out = 1 - seg(t, t1 - 0.5, t1)
    if t < t0 or a_out <= 0: return
    fb = font(F_BOLD, 130); big = 'MK-8'
    x = (W - fb.getlength(big)) / 2
    for i, ch in enumerate(big):
        k = ease_out(seg(t, t0 + i * 0.08, t0 + i * 0.08 + 0.45))
        d.text((x, y - 60 * (1 - k)), ch, font=fb, fill=WHITE + (int(255 * k * a_out),))
        x += fb.getlength(ch)
    k = ease_out(seg(t, t0 + 0.4, t0 + 1.0))
    ln = 380 * k
    d.line([(W / 2 - ln / 2, y + 160), (W / 2 + ln / 2, y + 160)], fill=CYAN + (int(255 * a_out),), width=3)
    fs = font(F_SEMI, 42)
    for j, s in enumerate(('Automatic Cigarette', 'Making Machine')):
        k = ease_out(seg(t, t0 + 0.6 + j * 0.15, t0 + 1.2 + j * 0.15))
        d.text(((W - fs.getlength(s)) / 2 + 40 * (1 - k), y + 185 + j * 54), s, font=fs,
               fill=WHITE + (int(240 * k * a_out),))

# ---------------------------------------------------------------- exploded parts (photo 2)
EXP_C = (980, 560)
Z0, Z1 = 0.68, 0.56
# name, rect, exploded target (canvas @Z1), label offset, align, delay, depth (px toward camera = -)
PARTS = [
    ('Fluid Reservoir',             (708, 30, 838, 192),    (205, 330),  (70, -40),   'left', 0.00, -260),
    ('Drive Motor',                 (1045, 122, 1245, 360), (545, 330),  (-45, -112), 'left', 0.10, -180),
    ('Material Feeding Hopper',     (1238, 82, 1642, 352),  (850, 470),  (-150, -120), 'left', 0.05, -120),
    ('Electrical Control Panel',    (630, 238, 990, 392),   (300, 560),  (-230, 75),  'left', 0.15, -320),
    ('Air Regulator & Valves',      (738, 398, 862, 502),   (115, 760),  (-80, -75),  'left', 0.30, -380),
    ('Conveyor & Transfer Assembly', (985, 338, 1692, 522), (690, 670),  (-330, 40),  'left', 0.20, -200),
    ('Side Housing',                (325, 402, 500, 692),   (110, 1430), (-70, 112),  'left', 0.35, -300),
    ('Rotary Drum Assembly',        (500, 518, 738, 722),   (350, 1420), (-115, -95), 'left', 0.30, -420),
    ('Roller & Gear Assembly',      (738, 556, 1128, 802),  (700, 1440), (-110, -105), 'left', 0.25, -360),
    ('Protective Guard',            (1262, 478, 1628, 642), (945, 1240), (100, -60),  'right', 0.40, -240),
    ('Collection Tray',             (308, 752, 668, 918),   (210, 1660), (-120, 70),  'left', 0.45, -260),
    ('Adjustment Bar & Dials',      (1272, 712, 1618, 802), (880, 1620), (-110, -42), 'left', 0.50, -300),
    ('Perforated Tray',             (1270, 818, 1628, 908), (880, 1745), (-90, 50),   'left', 0.55, -200),
    ('Sensor / Switch Bracket',     (1010, 818, 1142, 932), (580, 1680), (-120, 75),  'left', 0.60, -340),
    ('Handwheel',                   (632, 876, 722, 958),   (470, 1570), (40, 0),     'left', 0.62, -420),
    ('Ventilated Housing',          (222, 818, 382, 972),   (110, 1820), (60, 0),     'left', 0.65, -180),
]
P = [dict(name=a, rect=b, tgt=c, lo=d, al=e, dl=f, dz=g, idx=i) for i, (a, b, c, d, e, f, g) in enumerate(PARTS)]

def holes_layer(src):
    hole = src.copy()
    rec = Image.fromarray((np.asarray(src).astype(np.float32) * 0.18 + np.array([4, 10, 18])).clip(0, 255).astype(np.uint8))
    m = Image.new('L', src.size, 0); md = ImageDraw.Draw(m)
    for p in P: md.rectangle(p['rect'], fill=255)
    hole.paste(rec, (0, 0), m)
    hd = ImageDraw.Draw(hole)
    for p in P: hd.rectangle(p['rect'], outline=CYAN, width=3)
    return hole

def explode_k(p, t):
    if t < T_ASM0 - 0.01:
        st = T_EXP0 + p['dl'] * (T_EXP1 - T_EXP0 - PART_DUR) / 0.65
        return ease(seg(t, st, st + PART_DUR))
    st = T_ASM0 + (0.65 - p['dl']) * (T_ASM1 - T_ASM0 - PART_DUR) / 0.65
    return 1 - ease(seg(t, st, st + PART_DUR))

def asm_end(p):
    return T_ASM0 + (0.65 - p['dl']) * (T_ASM1 - T_ASM0 - PART_DUR) / 0.65 + PART_DUR

def exp_cam(t):
    if t < T_ASM0:
        z = lerp(Z0, Z1, ease(seg(t, T_EXP0 - 0.3, T_EXP1)))
    else:
        z = lerp(Z1, Z0, ease(seg(t, T_ASM0, T_ASM1 + 0.3)))
    # orbit: gentle swing, wider while exploded
    amp = 6 + 9 * ease(seg(t, T_EXP0, T_EXP1)) * (1 - ease(seg(t, T_ASM0, T_ASM1)))
    yaw = amp * math.sin((t - T_SCAN0) * 0.55)
    return Cam(EXP_C[0], EXP_C[1], z, yaw)

def part_world(p, t, cam, k):
    """World-space centre (X,Y,Z) and own spin (deg) of a part."""
    x0, y0, x1, y1 = p['rect']
    rx, ry = cam.world((x0 + x1) / 2, (y0 + y1) / 2)
    tx = (p['tgt'][0] - W / 2) * cam.z / Z1
    ty = (p['tgt'][1] - H / 2) * cam.z / Z1
    X, Y = lerp(rx, tx, k), lerp(ry, ty, k)
    Z = p['dz'] * cam.z / Z1 * math.sin(math.pi * k) * 0.6 + p['dz'] * 0.22 * k * cam.z / Z1
    ph = p['idx'] * 1.7
    if k > 0.99:
        Y += 5 * math.sin(t * 1.6 + ph)
    spin = (35 if p['idx'] % 2 else -35) * math.sin(math.pi * k) + k * 10 * math.sin(t * 0.9 + ph)
    return X, Y, Z, spin

def card_quad(cam, X, Y, Z, w, h, spin, back=0.0):
    sp = math.radians(spin); c, s = math.cos(sp), math.sin(sp)
    pts = []
    for (u, v) in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append(cam.proj(X + u * c - back * s, Y + v, Z + u * s + back * c))
    return pts

def avg_edge_color(sp):
    a = np.asarray(sp.convert('RGB')).astype(np.float32)
    return tuple(int(v * 0.45) for v in a.reshape(-1, 3).mean(0))

def render_exploded(t, show_labels):
    cam = exp_cam(t)
    src = anim_src(2, t)
    ks = [explode_k(p, t) for p in P]
    kmax = max(ks)
    frame = STUDIO.copy().convert('RGBA')
    base = src
    if kmax > 0:
        base = Image.blend(src, holes_layer(src), clamp(kmax * 3))
    frame.alpha_composite(plate(rgba(base), cam))
    # wireframe residue on frame during exploded state
    shadow = Image.new('L', (W, H), 0); sd = ImageDraw.Draw(shadow)
    fx = Image.new('RGBA', (W, H), (0, 0, 0, 0)); fd = ImageDraw.Draw(fx)
    items = []
    for p, k in zip(P, ks):
        X, Y, Z, spin = part_world(p, t, cam, k)
        x0, y0, x1, y1 = p['rect']
        w, h = (x1 - x0) * cam.z, (y1 - y0) * cam.z
        zc = -X * cam.s + Z * cam.c
        items.append((zc, p, k, X, Y, Z, spin, w, h))
    items.sort(key=lambda it: -it[0])   # far first
    for zc, p, k, X, Y, Z, spin, w, h in items:
        if k > 0.01:
            qd = [(a, b) for a, b, _ in card_quad(cam, X, Y, Z, w, h, spin)]
            sd.polygon([(a + 18 * k, b + 34 * k) for a, b in qd], fill=int(150 * min(1, k * 2)))
    if kmax > 0.01:
        shl = Image.new('RGBA', (W, H), (0, 0, 0, 0)); shl.putalpha(shadow.filter(ImageFilter.GaussianBlur(14)))
        frame.alpha_composite(shl)
    labels_pos = {}
    for zc, p, k, X, Y, Z, spin, w, h in items:
        sp_img = src.crop(p['rect']).convert('RGBA')
        q = card_quad(cam, X, Y, Z, w, h, spin)
        qd = [(a, b) for a, b, _ in q]
        cxs, cys, _ = cam.proj(X, Y, Z)
        labels_pos[p['idx']] = (cxs, cys, q)
        if k > 0.01:
            # thickness: back face edge strip
            T = 26 * cam.z / Z1
            qb = card_quad(cam, X, Y, Z, w, h, spin, back=T)
            qbd = [(a, b) for a, b, _ in qb]
            col = avg_edge_color(sp_img) + (int(255 * min(1, k * 3)),)
            side = [qd[1], qbd[1], qbd[2], qd[2]] if (spin + math.degrees(cam.yaw)) < 0 else [qd[0], qbd[0], qbd[3], qd[3]]
            top = [qd[0], qd[1], qbd[1], qbd[0]]
            layer = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
            ld.polygon(top, fill=tuple(min(255, int(c * 1.5)) for c in col[:3]) + (col[3],))
            ld.polygon(side, fill=col)
            frame.alpha_composite(layer)
        xs = [a for a, _ in qd]; ys = [b for _, b in qd]
        bx0, by0 = int(math.floor(min(xs))) - 2, int(math.floor(min(ys))) - 2
        bx1, by1 = int(math.ceil(max(xs))) + 2, int(math.ceil(max(ys))) + 2
        bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
        sw, shh = sp_img.size
        card = warp_plane(sp_img, [(0, 0), (sw, 0), (sw, shh), (0, shh)], qd, (bw, bh), (bx0, by0))
        frame.alpha_composite(card, (bx0, by0)) if (bx0 >= 0 and by0 >= 0 and bx1 <= W and by1 <= H) else _safe_paste(frame, card, bx0, by0)
        if k > 0.02:
            a = int(160 * min(1, k * 2))
            fd.line(qd + [qd[0]], fill=CYAN + (a,), width=2)
        # trails
        if 0.03 < k < 0.97:
            pts = []
            for j in range(10):
                tt = t - j * 0.035
                cc = exp_cam(tt); kk = explode_k(p, tt)
                Xj, Yj, Zj, _ = part_world(p, tt, cc, kk)
                pts.append(cc.proj(Xj, Yj, Zj)[:2])
            for j in range(9):
                fd.line([pts[j], pts[j + 1]], fill=CYAN + (int(220 * (1 - j / 9)),), width=max(1, 6 - j // 2))
        if t >= T_ASM0:
            dt = t - asm_end(p)
            if 0 <= dt < 0.5:
                qq = dt / 0.5; rr = 20 + 90 * ease_out(qq); a = int(255 * (1 - qq))
                fd.ellipse((cxs - rr, cys - rr, cxs + rr, cys + rr), outline=CYAN + (a,), width=4)
                fd.line(qd + [qd[0]], fill=WHITE + (a,), width=3)
    frame.alpha_composite(fx.filter(ImageFilter.GaussianBlur(6)))
    frame.alpha_composite(fx)
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    if show_labels > 0:
        for p in P:
            k = ks[p['idx']]
            if k < 0.98: continue
            cxs, cys, _ = labels_pos[p['idx']]
            lk = seg(t, T_LAB0 + p['idx'] * 0.3, T_LAB0 + p['idx'] * 0.3 + 0.6) * show_labels
            label(d, (cxs, cys), (cxs + p['lo'][0], cys + p['lo'][1]), p['name'], lk, p['al'])
        n = len(P)
        fxp, fyp, _ = cam.proj(*cam.world(1150, 960))
        lk = seg(t, T_LAB0 + n * 0.3, T_LAB0 + n * 0.3 + 0.6) * show_labels
        label(d, (fxp, fyp), (fxp - 300, fyp + 10), 'Main Machine Frame', lk)
    return frame, ov, d, None

def _safe_paste(frame, card, x, y):
    tmp = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    tmp.paste(card, (x, y), card)
    frame.alpha_composite(tmp)

def exploded_frame(t, show_labels):
    frame, ov, d, shl = render_exploded(t, show_labels)
    return frame, ov, d

# ---------------------------------------------------------------- wireframe scan
def scan_layer(i, cam, t, t0, t1, mask=FEATHER):
    k = seg(t, t0, t1)
    if k <= 0 or k >= 1: return None
    hgt = IMG[i].height
    yb = lerp(-150, hgt + 150, ease(k))
    ys = np.arange(hgt, dtype=np.float32)
    band = np.exp(-((ys - yb) / 70) ** 2) + 0.25 * (ys < yb) * np.exp(-(yb - ys) / 260)
    a = EDGES[i] * band[:, None]
    rgba_arr = np.zeros((hgt, IMG[i].width, 4), np.uint8)
    rgba_arr[..., 0], rgba_arr[..., 1], rgba_arr[..., 2] = CYAN
    rgba_arr[..., 3] = np.clip(a * 255 * 1.4, 0, 255).astype(np.uint8)
    line = np.exp(-((ys - yb) / 4) ** 2)
    rgba_arr[..., 3] = np.maximum(rgba_arr[..., 3], (line[:, None] * 200 * np.ones((1, IMG[i].width))).astype(np.uint8))
    img = Image.fromarray(rgba_arr, 'RGBA')
    lay = plate(img, cam)
    return lay

# ---------------------------------------------------------------- close-ups
CLOSE = {
    4: ((470, 560, 2.0, -6), (1480, 600, 2.0, 6), [
        ((235, 160), 'Pneumatic Solenoid Valves', 'right'), ((565, 650), 'Grinding Wheel', 'right'),
        ((465, 945), 'Pulley', 'right'), ((925, 600), 'Rotary Drum', 'left'), ((955, 970), 'Gears', 'right'),
        ((1350, 720), 'Toothed Transfer Drum', 'left'), ((1480, 340), 'Limit Switch', 'left'),
        ((1700, 330), 'Pneumatic Hoses', 'left'), ((1817, 835), 'Drum Assembly', 'left')]),
    3: ((300, 580, 1.78, 5), (1250, 560, 1.9, -5), [
        ((280, 370), 'Main Power Isolator', 'right'), ((285, 505), 'Drum Temperature Display', 'right'),
        ((290, 700), 'Gum / Heater Controls', 'right'), ((290, 900), 'Knife / Filter Selectors', 'right'),
        ((817, 316), 'Pulley', 'right'), ((790, 650), 'Transparent Safety Guard', 'right'),
        ((1166, 404), 'Gear Train', 'left'), ((1135, 850), 'Hand Lever', 'left')]),
    5: ((1000, 470, 1.62, -4), (1000, 560, 1.78, 4), [
        ((990, 160), 'Protective Cover', 'right'), ((990, 545), 'Adjustment Bar & Dials', 'right'),
        ((990, 710), 'Perforated Tray', 'right')]),
}

YY = np.linspace(0, 1, H)
DOF = Image.fromarray((np.repeat(np.clip(np.maximum((0.22 - YY) / 0.18, (YY - 0.78) / 0.18), 0, 1)[:, None], W, 1) * 255).astype(np.uint8))

def render_close(t, t0, t1, i):
    a, b, labs = CLOSE[i]
    k = ease(seg(t, t0, t1))
    cx, cy, s, yaw = (lerp(a[j], b[j], k) for j in range(4))
    cam = Cam(cx, cy, s, yaw)
    frame = BG[i].copy().convert('RGBA')
    frame.alpha_composite(plate(rgba(anim_src(i, t)), cam))
    if i == 5:   # scanning beam over tray
        lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        yb = lerp(500, 1450, (t - t0) / (t1 - t0))
        ld.rectangle((0, yb - 2, W, yb + 2), fill=CYAN + (200,))
        frame.alpha_composite(lay.filter(ImageFilter.GaussianBlur(5))); frame.alpha_composite(lay)
    rgb = frame.convert('RGB')
    rgb.paste(rgb.filter(ImageFilter.GaussianBlur(9)), (0, 0), DOF)
    frame = rgb.convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    for (sx, sy), text, side in labs:
        X, Y, _ = cam.proj(*cam.world(sx, sy))
        vis = clamp(1 - max(abs(X - W / 2) - 400, 0) / 90) * clamp(1 - max(abs(Y - H / 2) - 700, 0) / 100)
        if vis <= 0: continue
        bx = X + 60 if side == 'right' else X - 60
        label(d, (X, Y), (bx, Y - 50), text, vis, 'left' if side == 'right' else 'right')
    return frame, ov, d

# ---------------------------------------------------------------- photo 1 (line) shots
def line_frame(t):
    if t < T_SCAN0 + 0.01:
        k = ease(seg(t, 0.0, 4.0))
        cam = Cam(lerp(1000, 1300, k), lerp(560, 650, k), lerp(0.56, 1.75, k), lerp(-10, 6, k))
        par = -60 * k
    else:
        k = ease(seg(t, T_HERO, DUR - 0.8))
        cam = Cam(lerp(1260, 960, k), lerp(660, 600, k), lerp(1.9, 0.6, k), lerp(8, -6, k))
        par = 60 * k
    frame = Image.fromarray(np.roll(np.asarray(BG[1]), int(par), axis=1)).convert('RGBA')
    frame.alpha_composite(plate(rgba(IMG[1]), cam))
    return frame, cam

# ---------------------------------------------------------------- glitch
def glitch(img, t):
    for c in CUTS:
        dt = t - c
        if -0.1 <= dt <= 0.13:
            q = 1 - abs(dt) / 0.13
            a = np.asarray(img).copy()
            rng = np.random.default_rng(int(t * 1000))
            sh = int(24 * q)
            a[..., 0] = np.roll(a[..., 0], sh, 1); a[..., 2] = np.roll(a[..., 2], -sh, 1)
            for _ in range(int(10 * q)):
                y = rng.integers(0, H - 80); hh = rng.integers(8, 70); dx = int(rng.integers(-80, 80) * q)
                a[y:y + hh] = np.roll(a[y:y + hh], dx, 1)
            a[::4] = (a[::4] * 0.75).astype(np.uint8)
            if abs(dt) < 0.04:
                a = np.clip(a.astype(np.int16) + 90, 0, 255).astype(np.uint8)
            return Image.fromarray(a)
    return img

# ---------------------------------------------------------------- compositor
def render(fi):
    t = fi / FPS
    ov = None
    if t < T_SCAN0:
        frame, cam = line_frame(t)
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        title(d, t, 0.3, 3.6, 250)
        hud(d, seg(t, 2.8, 3.6), '')
    elif t < T_CLOSE[0][0] or (T_BACK <= t < T_HERO):
        if t < T_EXP0: cap = '> SCANNING ASSEMBLY  //  MACHINE RUNNING'
        elif t < T_EXP1: cap = '> EXPLODED VIEW  //  SEPARATING ASSEMBLIES'
        elif t < T_CLOSE[0][0]: cap = '> COMPONENT IDENTIFICATION'
        elif t < T_ASM1: cap = '> REASSEMBLY  //  PRECISION ALIGNMENT'
        else: cap = '> ASSEMBLY LOCKED  //  SYSTEM READY'
        sl = 0.0
        if T_LAB0 <= t < T_CLOSE[0][0]:
            sl = 1 - seg(t, T_CLOSE[0][0] - 0.4, T_CLOSE[0][0])
        frame, ov, d = exploded_frame(t, sl)
        cam = exp_cam(t)
        lay = scan_layer(2, cam, t, T_SCAN0 + 0.2, T_SCAN1 + 0.2)
        if lay is not None: frame.alpha_composite(lay)
        lay = scan_layer(2, cam, t, T_ASM1 + 0.1, T_HERO + 0.1)
        if lay is not None: frame.alpha_composite(lay)
        hud(d, 1.0, cap)
    elif t < T_BACK:
        for (t0, t1, i) in T_CLOSE:
            if t0 <= t < t1: break
        frame, ov, d = render_close(t, t0, t1, i)
        hud(d, 1.0, '> DETAIL INSPECTION  //  ' + {4: 'DRUMS & PULLEYS', 3: 'CONTROLS & DRIVE', 5: 'TRAY & GUARD'}[i])
    else:
        frame, cam = line_frame(t)
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        title(d, t, 33.6, DUR + 1, 230)
        a = seg(t, 34.4, 35.0)
        if a > 0:
            fm = font(F_MONO, 24); s = 'ENGINEERED  ·  RELIABLE  ·  PRODUCTION READY'
            d.text(((W - fm.getlength(s)) / 2, 1620), s, font=fm, fill=CYAN + (int(255 * a),))
        hud(d, 1 - seg(t, 32.5, 33.5), '')
    frame.alpha_composite(ov)
    out = frame.convert('RGB')
    if t > DUR - 0.5:
        out = Image.blend(out, Image.new('RGB', (W, H)), seg(t, DUR - 0.5, DUR))
    return glitch(out, t)

VIG = None
def finish(img, fi):
    global VIG
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        VIG = np.clip(1 - 0.35 * (((xx - W / 2) / (W * .7)) ** 2 + ((yy - H / 2) / (H * .7)) ** 2), 0.55, 1)[..., None]
    a = np.asarray(img).astype(np.float32) * VIG
    a += np.random.default_rng(fi).normal(0, 2.0, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def work(fi):
    finish(render(fi), fi).save(os.path.join(os.environ['FRAMES'], f'{fi:05d}.jpg'), quality=93)
    return fi

if __name__ == '__main__':
    if '--still' in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index('--still') + 1].split(',')]
        od = sys.argv[sys.argv.index('--still') + 2]
        for t in ts:
            finish(render(int(round(t * FPS))), 0).save(os.path.join(od, f'v2_{t:05.1f}.jpg'), quality=90)
        sys.exit()
    os.makedirs(os.environ['FRAMES'], exist_ok=True)
    n = int(DUR * FPS)
    with Pool(4) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, range(n), chunksize=6)):
            if i % 120 == 0: print(i, '/', n, flush=True)
