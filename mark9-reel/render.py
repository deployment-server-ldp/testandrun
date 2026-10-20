"""Mark 9 cinematic reveal — 1080x1920, 30 fps, 15 s.
Built only from real pixels of the reference photo (cutout bcut1.png)."""
import sys, subprocess, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter

W, H, FPS, DUR = 1080, 1920, 30, 15.0
NF = int(FPS * DUR)
FONT = "/usr/share/fonts/opentype/inter/"

# ---------------------------------------------------------------- source prep
src = Image.open("bcut1.png").convert("RGBA")
a = np.array(src).astype(np.float32)
a[420:, 368:526, 3] = 0          # floor air-hose coil + floor smudge
a[:252, :38, 3] = 0              # wall object at far left
a[:, 1125:, 3] = 0
a[..., 3] = np.where(a[..., 3] < 12, 0, a[..., 3])
SRC = a                            # HxWx4 float
SH, SW = SRC.shape[:2]
BASE_Y = 565                       # machine feet in source
ys, xs = np.nonzero(SRC[..., 3] > 30)
BX0, BX1 = xs.min(), xs.max()
MCX = (BX0 + BX1) / 2

# parts: name, rects, dx, dy, scale  (offsets in source px at full explode)
PARTS = [
    ("duct",        [(550, 18, 834, 64), (555, 64, 600, 300)], 0, -440, 1.00),
    ("hopper",      [(834, 18, 1012, 216), (1012, 18, 1125, 64)], 0, -335, 1.03),
    ("beacon_box",  [(155, 86, 245, 186)],                 0, -330, 1.00),
    ("bobbin",      [(678, 118, 806, 247)],               10, -250, 1.10),
    ("ctrl_panel",  [(596, 90, 676, 264)],               -45, -190, 1.05),
    ("feeder_hood", [(112, 184, 400, 252)],                0, -200, 1.02),
    ("feeder_win",  [(88, 252, 276, 348)],                 0, -95, 1.08),
    ("guard",       [(596, 314, 1006, 393)],               0, 70, 1.07),
    ("feeder_door", [(92, 384, 278, 493)],                 0, 120, 1.06),
    ("cabinets",    [(602, 457, 1002, 566)],               0, 150, 1.02),
]
PAD_T, PAD_B, PAD_X = 520, 330, 90
CW, CH = SW + 2 * PAD_X, SH + PAD_T + PAD_B

core = SRC.copy()
part_imgs = []
taken = np.zeros((SH, SW), bool)
for name, rects, dx, dy, sc in PARTS:
    m = np.zeros((SH, SW), bool)
    for (x0, y0, x1, y1) in rects:
        m[y0:y1, x0:x1] = True
    m &= ~taken
    taken |= m
    img = np.zeros_like(SRC)
    img[m] = SRC[m]
    yy, xx = np.nonzero(m & (SRC[..., 3] > 0))
    bb = (xx.min(), yy.min(), xx.max() + 1, yy.max() + 1)
    crop = Image.fromarray(img[bb[1]:bb[3], bb[0]:bb[2]].astype(np.uint8), "RGBA")
    if sc != 1:
        crop = crop.resize((round(crop.width * sc), round(crop.height * sc)), Image.LANCZOS)
    part_imgs.append((crop, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2), dx, dy))
# parts mounted on top leave empty space; cover panels reveal a plain dark back plate
EXTERNAL = {"duct", "hopper", "beacon_box"}
for (name, rects, *_), in zip(PARTS):
    m = np.zeros((SH, SW), bool)
    for (x0, y0, x1, y1) in rects:
        m[y0:y1, x0:x1] = True
    m &= taken
    if name in EXTERNAL:
        core[m, 3] = 0
    else:
        n = np.random.default_rng(1).normal(0, 1.5, (m.sum(), 1))
        core[m, :3] = np.array([16, 17, 20]) + n
core[..., 3] = gaussian_filter(core[..., 3], 0.6)
core_img = Image.fromarray(core.astype(np.uint8), "RGBA")
full_img = Image.fromarray(SRC.astype(np.uint8), "RGBA")

# ---------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def seg(t, t0, t1): return clamp((t - t0) / (t1 - t0))
def ease(x): return x * x * (3 - 2 * x)
def eio(x): return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def eout(x): return 1 - (1 - x) ** 4
def lerp(a, b, x): return a + (b - a) * x

def persp_coeffs(dst, srcpts):
    m = []
    for (x, y), (u, v) in zip(dst, srcpts):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    return np.linalg.solve(np.array(m, float), np.array(srcpts, float).reshape(8))

# ---------------------------------------------------------------- timeline
TL_EXP = [(6.15 + 0.11 * i, 2.1) for i in range(len(PARTS))]
LOCK = 11.55

def explode_p(t, i):
    if t < 10.0:
        t0, d = TL_EXP[i]
        return eio(seg(t, t0, t0 + d))
    t0 = 10.0 + 0.07 * (len(PARTS) - 1 - i)
    return 1 - eio(seg(t, t0, LOCK))

def camera(t):
    """scale, centre-x, floor-y, lift(px src), orbit k, keystone q"""
    if t < 3:
        s = lerp(0.66, 0.835, eout(seg(t, 0, 1.5))) + 0.012 * seg(t, 1.5, 3)
        return s, 540, 1170, 0, 0, 0
    if t < 6:
        x = seg(t, 3, 6)
        lift = 90 * eio(seg(t, 3.0, 5.0)) + 4 * math.sin((t - 5) * 3) * seg(t, 5, 5.4)
        return lerp(0.847, 0.86, x), lerp(540, 528, eio(x)), 1170, lift, 0.055 * math.sin(math.pi * x), 0
    if t < 12:
        x = seg(t, 6, 10)
        lift = 90 + 4 * math.sin((t - 5) * 3)
        dn = eio(seg(t, 6.0, 7.4)) * (1 - eio(seg(t, 10.0, LOCK)))
        lift += 70 * dn
        s = lerp(0.86, 0.875, eio(x))
        s += 0.006 * math.exp(-max(0, t - LOCK) * 9) * (t >= LOCK)
        return s, lerp(528, 540, eio(seg(t, 6, 7))), 1170 + 80 * dn, lift, 0, 0
    x = seg(t, 12, 15)
    d = seg(t, 12.0, 13.15)
    lift = (90 + 4 * math.sin((t - 5) * 3)) * (1 - eio(d))
    return lerp(0.875, 0.94, eout(x)), 540, lerp(1170, 1390, eio(seg(t, 12, 13.6))), lift, 0, 0.028 * eio(seg(t, 12, 14))

# ---------------------------------------------------------------- static layers
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def background(floor_y):
    r = np.sqrt(((xx - 540) / 900) ** 2 + ((yy - floor_y + 250) / 1100) ** 2)
    base = 26 - 18 * np.clip(r, 0, 1)
    bg = np.stack([base * 0.92, base * 0.96, base * 1.06], -1)
    f = yy > floor_y - 40
    fl = np.clip((yy - (floor_y - 40)) / 60, 0, 1)
    floor = (8 + 10 * np.exp(-((xx - 540) / 520) ** 2) * np.exp(-((yy - floor_y) / 320) ** 2))
    bg = bg * (1 - fl[..., None]) + np.stack([floor * .95, floor, floor * 1.08], -1) * fl[..., None]
    return bg
BG_CACHE = {}
def bg_for(fy):
    k = int(round(fy / 2) * 2)
    if k not in BG_CACHE:
        BG_CACHE[k] = background(k)
    return BG_CACHE[k]

grain_bank = [np.random.default_rng(i).normal(0, 2.2, (H, W, 1)).astype(np.float32) for i in range(6)]
vign = (1 - 0.35 * np.clip(np.sqrt(((xx - 540) / 760) ** 2 + ((yy - 960) / 1150) ** 2) - 0.45, 0, 1))[..., None]

def font(name, size): return ImageFont.truetype(FONT + name, size)
def tracked(draw_img, text, fnt, cy, track, alpha, color=(255, 255, 255)):
    widths = [fnt.getlength(c) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = (W - total) / 2
    d = ImageDraw.Draw(draw_img)
    for c, w in zip(text, widths):
        d.text((x, cy), c, font=fnt, fill=color + (int(255 * alpha),), anchor="lm")
        x += w + track
    return (W - total) / 2, total

F_HOOK = font("InterDisplay-Light.otf", 50)
F_MARK = font("InterDisplay-Bold.otf", 190)
F_SUB = font("InterDisplay-Medium.otf", 40)
F_TAG = font("Inter-Light.otf", 23)

# ---------------------------------------------------------------- frame
def machine_canvas(t):
    can = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    if t < 6.1 or t >= LOCK + 0.001:
        can.alpha_composite(full_img, (PAD_X, PAD_T))
        return can, []
    can.alpha_composite(core_img, (PAD_X, PAD_T))
    leaders = []
    for i, (img, (cx, cy), dx, dy) in enumerate(part_imgs):
        p = explode_p(t, i)
        drift = 3 * math.sin(t * 2.1 + i * 1.7) * min(p, 1)
        sc = lerp(1.0, 1.0, p)
        im = img
        if abs(p - 1) > 1e-3 and img.size != (0, 0):
            # scale grows with p from 1 -> part scale
            base_w = img.width
            orig = part_imgs[i][0]
        nx, ny = cx + dx * p, cy + dy * p + drift
        # interpolate scale toward stored scaled size
        f = lerp(1 / PARTS[i][4], 1.0, p)
        if abs(f - 1) > 1e-3:
            im = img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.BILINEAR)
        can.alpha_composite(im, (int(round(PAD_X + nx - im.width / 2)), int(round(PAD_T + ny - im.height / 2))))
        if p > 0.02:
            leaders.append(((cx, cy), (nx, ny), p))
    return can, leaders

def render(fi):
    t = fi / FPS
    s, cxf, floor_y, lift, k, q = camera(t)
    can, leaders = machine_canvas(t)
    # dashed CAD leader lines
    if leaders:
        ld = ImageDraw.Draw(can)
        for (x0, y0), (x1, y1), p in leaders:
            L = math.hypot(x1 - x0, y1 - y0)
            n = int(L // 14)
            al = int(70 * min(1, p * 1.5))
            for j in range(n):
                a0, a1 = j / n, (j + 0.5) / n
                ld.line([(PAD_X + x0 + (x1 - x0) * a0, PAD_T + y0 + (y1 - y0) * a0),
                         (PAD_X + x0 + (x1 - x0) * a1, PAD_T + y0 + (y1 - y0) * a1)], fill=(170, 200, 255, al), width=2)
    # scale
    m = can.resize((round(CW * s), round(CH * s)), Image.LANCZOS)
    mw, mh = m.size
    # orbit / keystone via perspective warp (bottom edge stays on the floor)
    if abs(k) > 1e-4 or q > 1e-4:
        kk = abs(k)
        cw = 1 - kk * 1.2
        if k >= 0:
            dst = [(mw * (1 - cw), kk * mh), (mw, 0), (mw, mh), (mw * (1 - cw), mh)]
        else:
            dst = [(0, 0), (mw * cw, kk * mh), (mw * cw, mh), (0, mh)]
        dst = [(x + (mw * q if y < mh / 2 and x < mw / 2 else -mw * q if y < mh / 2 else 0), y) for x, y in dst]
        co = persp_coeffs(dst, [(0, 0), (mw, 0), (mw, mh), (0, mh)])
        m = m.transform((mw, mh), Image.PERSPECTIVE, tuple(co), Image.BICUBIC)
        # keep visual centre stable
        cshift = (mw * (1 - cw) / 2) * (1 if k >= 0 else -1)
    else:
        cshift = 0
    M = np.array(m).astype(np.float32)
    rgb, al = M[..., :3], M[..., 3:] / 255.0
    # rim light from top-left
    sh = np.zeros_like(al); sh[3:, 2:] = al[:-3, :-2]
    rim = np.clip(al - sh, 0, 1)
    flash = 0.9 * math.exp(-max(0, t - LOCK) * 7) if t >= LOCK else 0
    rgb = rgb + rim * np.array([150, 175, 215]) * (0.45 + flash)
    # position in frame
    ox = int(round(cxf - (PAD_X + MCX) * s + cshift))
    oy = int(round(floor_y - lift * s - (PAD_T + BASE_Y) * s))
    frame = bg_for(floor_y).copy()

    # reveal light (scene 1)
    if t < 3.2:
        bx = lerp(-150, 1250, ease(seg(t, 0.25, 2.4)))
        glob = 0.035 + 0.965 * ease(seg(t, 1.5, 3.1))
        colx = np.arange(mw) + ox
        beam = np.exp(-((colx - bx) / 85) ** 2)
        vis = np.clip(glob + 1.1 * beam * (t < 2.9), 0, 1.25)[None, :, None]
        rgb = rgb * vis
        frame *= glob ** 0.6 * 0.85 + 0.15
    # light sweep (scene 5)
    sweep = 0
    if t > 13.1:
        p = seg(t, 13.2, 14.4)
        if 0 < p < 1:
            cyy, cxx = np.mgrid[0:mh, 0:mw]
            band = lerp(-400, mw + 400, ease(p))
            sweep = np.exp(-(((cxx + (cyy * 0.45)) - band) / 70) ** 2)[..., None] * 0.55
            rgb = rgb + sweep * 255 * al
    # reflection + shadow
    rh = int(mh)
    refl_img = rgb[::-1] ; refl_a = al[::-1]
    machine_bottom = oy + (PAD_T + BASE_Y) * s
    gap = 2 * lift * s
    rtop = int(round(machine_bottom + gap - (mh - (PAD_T + BASE_Y) * s)))
    fade = np.clip(1 - (np.arange(mh)[:, None, None] - (mh - (PAD_T + BASE_Y) * s)) / 260, 0, 1) * 0.22
    def paste(dst, img, alpha, x, y):
        h_, w_ = img.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w_), min(H, y + h_)
        if x1 <= x0 or y1 <= y0: return
        sub = img[y0 - y:y1 - y, x0 - x:x1 - x]; sa = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
        dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - sa) + sub * sa
    # contact shadow
    sw_ = (BX1 - BX0) * s
    hn = lift / 90
    shadow = np.exp(-((xx - cxf) / (sw_ * 0.52)) ** 8) * np.exp(-((yy - floor_y - 6) / (14 + 30 * hn)) ** 2) * (0.85 - 0.45 * hn)
    if t < 3.2: shadow *= ease(seg(t, 1.2, 3.0))
    frame *= (1 - shadow[..., None])
    rf = gaussian_filter(refl_img, sigma=(1.5, 1.5, 0)) if True else refl_img
    paste(frame, rf * 0.9, refl_a * fade, ox, rtop)
    paste(frame, rgb, al, ox, oy)

    # beam shaft (scene 1)
    if 0.15 < t < 3.0:
        bx = lerp(-150, 1250, ease(seg(t, 0.25, 2.4)))
        tilt = (yy - 0) * 0.08
        shaft = np.exp(-((xx - bx + 140 - tilt) / 24) ** 2) * (yy < floor_y + 10) * 0.10 * (1 - seg(t, 2.4, 2.95))
        hot = np.exp(-((xx - bx) / 120) ** 2) * np.exp(-((yy - floor_y - 15) / 18) ** 2) * 0.35 * (1 - seg(t, 2.4, 2.95))
        frame += ((shaft + hot)[..., None] * np.array([210, 225, 255]))

    # text
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if 0.9 < t < 3.05:
        a1 = ease(seg(t, 0.95, 1.55)) * (1 - ease(seg(t, 2.65, 3.0)))
        tracked(txt, "ENGINEERING IN MOTION", F_HOOK, 560, lerp(10, 17, eout(seg(t, 0.95, 3.0))), a1)
    if t > 12.7:
        a_m = ease(seg(t, 12.75, 13.35))
        rise = (1 - eout(seg(t, 12.75, 13.6))) * 30
        x0, tw = tracked(txt, "MARK 9", F_MARK, 470 + rise, lerp(30, 14, eout(seg(t, 12.75, 14))), a_m)
        a_s = ease(seg(t, 13.25, 13.8))
        tracked(txt, "CIGARETTE MAKING MACHINE", F_SUB, 635 + (1 - eout(seg(t, 13.25, 13.9))) * 20, 9, a_s)
        lw = 560 * eout(seg(t, 13.6, 14.3))
        if lw > 1:
            ImageDraw.Draw(txt).line([(540 - lw / 2, 695), (540 + lw / 2, 695)], fill=(170, 195, 235, int(200 * a_s)), width=2)
        a_t = ease(seg(t, 13.8, 14.35))
        tracked(txt, "PRECISION ENGINEERING.  INDUSTRIAL PERFORMANCE.", F_TAG, 740, 3.2, a_t * 0.85, (215, 225, 240))
    T = np.array(txt).astype(np.float32)
    ta = T[..., 3:] / 255
    # light sweep across title
    if t > 13.6:
        p = seg(t, 13.7, 14.6)
        if 0 < p < 1:
            band = lerp(-200, W + 200, ease(p))
            glow = np.exp(-(((xx + yy * 0.35) - band - 180) / 50) ** 2)[..., None]
            T[..., :3] = T[..., :3] * (0.93 + 0.07 * glow) + glow * np.array([0, 10, 30]) * (T[..., :3] > 0)
    frame = frame * (1 - ta) + T[..., :3] * ta

    # grade
    frame = frame * vign
    frame = frame + grain_bank[fi % 6]
    if t > 14.7: frame *= 1  # hold
    lum = frame.mean(-1, keepdims=True)
    frame = lum + (frame - lum) * 1.06
    frame = np.clip(frame, 0, 255)
    return frame.astype(np.uint8)

if __name__ == "__main__":
    if sys.argv[1] == "stills":
        for ts in sys.argv[2:]:
            Image.fromarray(render(int(float(ts) * FPS))).save(f"still_{ts}.jpg", quality=90)
    else:
        out = sys.argv[2]
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                              "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        for fi in range(NF):
            p.stdin.write(render(fi).tobytes())
            if fi % 30 == 0: print(fi, flush=True)
        p.stdin.close(); p.wait()
