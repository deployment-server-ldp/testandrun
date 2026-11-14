"""MK-8 reel v3 in the house layout of the Civic KDF-2 reference reel.

Header (brand / model / subtitle), animated 16:9 machine band in the middle with
labels, glow rings and pop-out parts, section number + typewriter title + 4-step
progress bar, footer. Animated content comes from the real reference photos
(rotating wheels via render2.anim_src).
"""
import math, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv; sys.argv = [sys.argv[0]]
import render2 as R2
sys.argv = _argv

W, H, FPS = 1080, 1920, 30
BW, BH, BY = 1080, 608, 636          # band size and top
YEL = (245, 178, 26)
WHITE = (255, 255, 255)
DUR = 25.0

F_BOLD = '/usr/share/fonts/opentype/inter/Inter-Bold.otf'
F_XB = '/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf'
F_SEMI = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
F_MED = '/usr/share/fonts/opentype/inter/Inter-Medium.otf'
F_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
for name in ('F_XB', 'F_SEMI', 'F_MED'):
    if not os.path.exists(globals()[name]): globals()[name] = F_BOLD
font = R2.font

BRAND = 'CIVIC TOBACCO MACHINERY'
MODEL = 'MK-8'
SUB = 'CIGARETTE MAKER'
FOOT = 'AUTOMATIC CIGARETTE MAKING MACHINE'

clamp, ease, ease_out, seg, lerp = R2.clamp, R2.ease, R2.ease_out, R2.seg, R2.lerp

# ---------------------------------------------------------------- shots
# label: (src x, y, text, (dx, dy) box offset from anchor in band px)
# pop:   (src rect, (dx, dy) band px)      ring: (src x, y, radius)
SECTIONS = [  # (num, title, t0, t1)
    (1, 'FEEDING & CONTROL', 1.0, 6.0),
    (2, 'DRUMS & ROLLERS', 6.0, 11.5),
    (3, 'TRAY & GUARDS', 11.5, 15.5),
    (4, 'DRIVE & CONTROL', 15.5, 20.5),
]
SHOTS = [
    dict(img=1, t0=0.0, t1=1.0, a=(1150, 640, 0.95), b=(1180, 640, 1.05), labels=[], pops=[], rings=[]),
    dict(img=2, t0=1.0, t1=6.0, a=(1060, 300, 1.08), b=(1110, 290, 1.22),
         labels=[(1440, 230, 'Material feeding hopper', (-40, 175)),
                 (1150, 250, 'Drive motor', (-60, -170)),
                 (773, 110, 'Fluid reservoir', (-330, -40)),
                 (810, 320, 'Electrical control panel', (-410, 120))],
         pops=[((1238, 82, 1642, 352), (40, -22)), ((1045, 122, 1245, 360), (0, -30)),
               ((708, 30, 838, 192), (-30, -10))],
         rings=[(1185, 215, 70)]),
    dict(img=2, t0=6.0, t1=8.6, a=(860, 650, 1.45), b=(900, 660, 1.6),
         labels=[(617, 620, 'Rotary drum assembly', (-170, -180)),
                 (930, 700, 'Roller & gear assembly', (60, 150)),
                 (800, 450, 'Air regulator & valves', (110, -60))],
         pops=[((738, 556, 1128, 802), (0, 30))],
         rings=[(867, 686, 30), (1010, 690, 30)]),
    dict(img=4, t0=8.6, t1=11.5, a=(820, 640, 0.95), b=(1100, 640, 1.08),
         labels=[(924, 601, 'Rotary drum', (-260, -190)),
                 (565, 652, 'Grinding wheel', (-200, -120)),
                 (1341, 706, 'Toothed transfer drum', (60, -230)),
                 (467, 946, 'Pulley', (150, 40)),
                 (955, 972, 'Gears', (90, 60))],
         pops=[], rings=[(924, 601, 190), (565, 652, 56)]),
    dict(img=5, t0=11.5, t1=15.5, a=(1000, 520, 1.0), b=(1000, 560, 1.12),
         labels=[(990, 170, 'Protective cover', (-430, 0)),
                 (990, 545, 'Adjustment bar & dials', (-470, -40)),
                 (990, 720, 'Perforated tray', (190, 40))],
         pops=[((690, 655, 1330, 805), (0, 45))], rings=[(1265, 528, 26), (705, 528, 26)]),
    dict(img=3, t0=15.5, t1=20.5, a=(700, 540, 1.0), b=(780, 540, 1.1),
         labels=[(280, 370, 'Main power isolator', (150, -80)),
                 (285, 505, 'Drum temperature display', (150, 15)),
                 (290, 760, 'Gum, heater & knife controls', (150, -15)),
                 (817, 316, 'Pulley', (120, 95)),
                 (1166, 404, 'Gear train', (-150, 175))],
         pops=[], rings=[(1110, 495, 80), (1166, 404, 60)]),
    dict(img=1, t0=20.5, t1=DUR, a=(1240, 650, 1.6), b=(980, 600, 0.6), labels=[], pops=[], rings=[]),
]
XF = 0.3  # crossfade

def band_shot(sh, t):
    """Render one shot into a 1080x608 band image + overlay drawing list."""
    i = sh['img']
    k = ease(seg(t, sh['t0'], sh['t1']))
    cx, cy, s = (lerp(sh['a'][j], sh['b'][j], k) for j in range(3))
    src = R2.anim_src(i, t)
    lt = t - sh['t0']; dur = sh['t1'] - sh['t0']
    pk = ease(seg(lt, 0.5, 1.3)) * (1 - ease(seg(lt, dur - 0.6, dur)))   # pop-out amount
    base = src
    if sh['pops'] and pk > 0:
        dark = Image.fromarray((np.asarray(src).astype(np.float32) * 0.25 + 8).clip(0, 255).astype(np.uint8))
        m = Image.new('L', src.size, 0); md = ImageDraw.Draw(m)
        for r, _ in sh['pops']: md.rectangle(r, fill=int(255 * min(1, pk * 2)))
        base = src.copy(); base.paste(dark, (0, 0), m)
    a = 1 / s
    img = base.transform((BW, BH), Image.AFFINE, (a, 0, cx - BW / 2 * a, 0, a, cy - BH / 2 * a), resample=Image.BICUBIC)
    img = img.convert('RGBA')
    def to_b(x, y): return BW / 2 + (x - cx) * s, BH / 2 + (y - cy) * s
    for r, (dx, dy) in sh['pops']:
        if pk <= 0: continue
        sp = src.crop(r)
        sc = s * (1 + 0.06 * pk)
        sw, shh = int((r[2] - r[0]) * sc), int((r[3] - r[1]) * sc)
        sp = sp.resize((max(1, sw), max(1, shh)), Image.BICUBIC).convert('RGBA')
        x, y = to_b((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        x += dx * pk; y += dy * pk
        shd = Image.new('RGBA', (sw + 60, shh + 60), (0, 0, 0, 0))
        ImageDraw.Draw(shd).rectangle((30, 30, 30 + sw, 30 + shh), fill=(0, 0, 0, int(140 * pk)))
        shd = shd.filter(ImageFilter.GaussianBlur(14))
        img.alpha_composite(shd, (int(x - sw / 2 - 30 + 12 * pk), int(y - shh / 2 - 30 + 22 * pk)))
        img.alpha_composite(sp, (int(x - sw / 2), int(y - shh / 2)))
    fx = Image.new('RGBA', (BW, BH), (0, 0, 0, 0)); fd = ImageDraw.Draw(fx)
    ov = Image.new('RGBA', (BW, BH), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    # glow rings
    for (x, y, r) in sh['rings']:
        rk = ease_out(seg(lt, 0.6, 1.2)) * (1 - seg(lt, dur - 0.4, dur))
        if rk <= 0: continue
        bx, by = to_b(x, y); rr = r * s * (1.02 + 0.03 * math.sin(t * 5))
        a = int(255 * rk)
        fd.ellipse((bx - rr, by - rr, bx + rr, by + rr), outline=YEL + (a,), width=6)
        od.ellipse((bx - rr, by - rr, bx + rr, by + rr), outline=(255, 236, 170, a), width=2)
    # labels
    for j, (x, y, text, (dx, dy)) in enumerate(sh['labels']):
        st = 0.5 + j * 0.45
        lk = seg(lt, st, st + 0.5) * (1 - seg(lt, dur - 0.35, dur))
        if lk <= 0: continue
        ax, ay = to_b(x, y)
        draw_label(od, fd, (ax, ay), (ax + dx, ay + dy), text, lk)
    img.alpha_composite(fx.filter(ImageFilter.GaussianBlur(7)))
    img.alpha_composite(fx)
    img.alpha_composite(ov)
    return img

def draw_label(od, fd, anchor, box, text, k):
    ax, ay = anchor; bx, by = box
    f = font(F_MED, 25)
    tw = f.getlength(text); bw, bh = tw + 34, 48
    right_of = bx > ax                        # box to the right of anchor -> bar on its left
    x0 = bx if right_of else bx - bw
    x0 = clamp(x0, 66, BW - 66 - bw); y0 = clamp(by - bh / 2, 40, BH - 40 - bh)
    ex = x0 if right_of else x0 + bw; ey = y0 + bh / 2
    kl = ease_out(k / 0.5)
    od.line([(ax, ay), (lerp(ax, ex, kl), lerp(ay, ey, kl))], fill=(255, 255, 255, 235), width=2)
    fd.ellipse((ax - 9, ay - 9, ax + 9, ay + 9), fill=WHITE + (int(200 * k),))
    od.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill=WHITE + (255,))
    kb = ease_out((k - 0.3) / 0.7)
    if kb <= 0: return
    slide = (1 - kb) * (24 if right_of else -24)
    x0 += slide; A = int(255 * kb)
    od.rectangle((x0 + 4, y0 + 5, x0 + bw + 4, y0 + bh + 5), fill=(0, 0, 0, int(60 * kb)))
    od.rectangle((x0, y0, x0 + bw, y0 + bh), fill=(244, 245, 246, int(242 * kb)))
    bar_x = x0 if right_of else x0 + bw - 5
    od.rectangle((bar_x, y0, bar_x + 5, y0 + bh), fill=YEL + (A,))
    od.text((x0 + 17, y0 + 10), text, font=f, fill=(28, 30, 34, A))

def band(t):
    cur = [sh for sh in SHOTS if sh['t0'] <= t < sh['t1'] + (XF if sh['t1'] < DUR else 0)]
    if not cur: cur = [SHOTS[-1]]
    img = band_shot(cur[0], t)
    if len(cur) > 1:
        q = ease(seg(t, cur[1]['t0'], cur[1]['t0'] + XF))
        img = Image.blend(img, band_shot(cur[1], t), q)
    return img

# ---------------------------------------------------------------- page layout
def spaced(d, xy, text, f, fill, sp):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill); x += f.getlength(ch) + sp
    return x

def page_bg(b):
    small = b.convert('RGB').resize((54, 30), Image.BILINEAR)
    s = max(W / 54, H / 30)
    big = small.resize((int(54 * s), int(30 * s)), Image.BILINEAR)
    l = (big.width - W) // 2
    big = big.crop((l, 0, l + W, H)).filter(ImageFilter.GaussianBlur(30))
    a = np.asarray(big).astype(np.float32)
    g = a.mean(-1, keepdims=True)
    a = (a * 0.25 + g * 0.75) * 0.42 + np.array([14, 16, 20])
    yy = np.linspace(0, 1, H)[:, None, None]
    a *= 1 - 0.35 * np.clip((yy - 0.6) / 0.4, 0, 1)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert('RGBA')

def section_at(t):
    for s in SECTIONS:
        if s[2] <= t < s[3]: return s
    return None

def render(fi):
    t = fi / FPS
    b = band(t)
    page = page_bg(b)
    d = ImageDraw.Draw(page)
    hk = ease_out(seg(t, 0.0, 0.8))
    A = int(255 * hk)
    # header
    spaced(d, (70, 252), BRAND, font(F_MONO, 26), YEL + (A,), 6)
    d.text((62, 290 + 20 * (1 - hk)), MODEL, font=font(F_XB, 158), fill=(246, 246, 244, A))
    spaced(d, (70, 482), SUB, font(F_SEMI, 54), (246, 246, 244, A), 5)
    d.line([(70, 568), (70 + 60 * hk, 568)], fill=YEL + (A,), width=3)
    spaced(d, (146, 553), 'EXPLODED VIEW  //  KEY COMPONENTS', font(F_MONO, 25), (210, 212, 215, A), 3)
    # band + frame
    page.alpha_composite(b, (0, BY))
    d.line([(0, BY - 1), (W, BY - 1)], fill=YEL + (255,), width=3)
    d.line([(0, BY + BH), (W, BY + BH)], fill=YEL + (255,), width=3)
    L, m = 34, 24
    for (x, y, sx, sy) in ((m, BY + m, 1, 1), (W - m, BY + m, -1, 1), (m, BY + BH - m, 1, -1), (W - m, BY + BH - m, -1, -1)):
        d.line([(x, y), (x + sx * L, y)], fill=(240, 240, 240, 230), width=3)
        d.line([(x, y), (x, y + sy * L)], fill=(240, 240, 240, 230), width=3)
    # section block
    sec = section_at(t)
    if sec:
        num, ttl, t0, t1 = sec
        k = ease_out(seg(t, t0, t0 + 0.35))
        d.text((66, 1338 + 30 * (1 - k)), f'{num:02d}', font=font(F_XB, 118), fill=YEL + (int(255 * k),))
        spaced(d, (237, 1362), 'SECTION', font(F_MONO, 25), (200, 202, 205, int(255 * k)), 5)
        n = int(len(ttl) * clamp((t - t0 - 0.1) / 0.7))
        tx = 237
        d.text((tx, 1395), ttl[:n], font=font(F_BOLD, 50), fill=WHITE + (255,))
        if n < len(ttl) or (t * 2) % 1 < 0.5:
            cx = tx + font(F_BOLD, 50).getlength(ttl[:n]) + 6
            d.rectangle((cx, 1402, cx + 14, 1448), fill=YEL + (255,))
        prog = [(1.0 if t >= s[3] else clamp((t - s[2]) / (s[3] - s[2]))) for s in SECTIONS]
    elif t >= SECTIONS[-1][3]:
        k = ease_out(seg(t, 20.5, 20.9))
        spaced(d, (70, 1365), 'OVERVIEW', font(F_MONO, 25), (200, 202, 205, int(255 * k)), 5)
        ttl = 'COMPLETE CIGARETTE MAKING LINE'
        n = int(len(ttl) * clamp((t - 20.6) / 0.9))
        d.text((70, 1400), ttl[:n], font=font(F_BOLD, 50), fill=WHITE + (255,))
        if n < len(ttl) or (t * 2) % 1 < 0.5:
            cx = 70 + font(F_BOLD, 50).getlength(ttl[:n]) + 6
            d.rectangle((cx, 1407, cx + 14, 1453), fill=YEL + (255,))
        prog = [1.0] * 4
    else:
        prog = [0.0] * 4
    # progress bar
    sx, gap, sw = 70, 24, (W - 140 - 3 * 24) / 4
    for j, p in enumerate(prog):
        x0 = sx + j * (sw + gap)
        d.rectangle((x0, 1520, x0 + sw, 1524), fill=(255, 255, 255, int(70 * hk)))
        if p > 0: d.rectangle((x0, 1520, x0 + sw * p, 1524), fill=YEL + (255,))
    spaced(d, (70, 1600), FOOT, font(F_MONO, 25), (150, 153, 158, A), 6)
    out = page.convert('RGB')
    if t > DUR - 0.4:
        out = Image.blend(out, Image.new('RGB', (W, H)), seg(t, DUR - 0.4, DUR))
    return out

def work(fi):
    render(fi).save(os.path.join(os.environ['FRAMES'], f'{fi:05d}.jpg'), quality=94)
    return fi

if __name__ == '__main__':
    if '--still' in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index('--still') + 1].split(',')]
        od = sys.argv[sys.argv.index('--still') + 2]
        for t in ts:
            render(int(round(t * FPS))).save(os.path.join(od, f'v3_{t:05.1f}.jpg'), quality=90)
        sys.exit()
    os.makedirs(os.environ['FRAMES'], exist_ok=True)
    n = int(DUR * FPS)
    with Pool(4) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, range(n), chunksize=6)):
            if i % 150 == 0: print(i, '/', n, flush=True)
