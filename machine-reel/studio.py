#!/usr/bin/env python3
"""Studio-style exploded view made directly from one machine photo (landscape 1920x1080).
Parts are cut from the photo (studio_parts.json), lift out, get labelled, the camera visits
both units, then everything reassembles.   python3 studio.py [--still 4.5 7 ...]"""
import json, math, os, subprocess, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
R = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1920, 1080, 30
TOTAL = 14.6
S0 = 0.80                                         # screen px per photo px at zoom 1
FD = "/usr/share/fonts/opentype/inter/"
TEAL = np.array([190, 170, 40], np.float32)       # BGR
# ---- timing
T_EXP, EXP_DUR, EXP_STAG = 1.6, 1.5, 0.085
T_LAB_OUT = 11.2
T_RE, RE_DUR, RE_STAG = 11.5, 1.3, 0.035
# camera keys: (time, centre x, centre y, zoom) in photo pixels
CAM = [(0, 900, 690, 1.22), (1.6, 900, 690, 1.15), (4.2, 975, 650, 1.0), (5.3, 975, 650, 1.0),
       (6.3, 500, 340, 1.5), (7.1, 520, 345, 1.54),          # close-up: left unit parts
       (8.1, 1500, 330, 1.5), (8.9, 1520, 335, 1.54),        # close-up: right unit top parts
       (9.9, 1200, 1010, 1.5), (10.7, 1220, 1005, 1.54),     # close-up: lower parts
       (11.8, 975, 650, 1.0), (13.6, 900, 690, 1.12), (14.6, 900, 690, 1.16)]

clamp = lambda v, a=0.0, b=1.0: max(a, min(b, v))
def sstep(a, b, v):
    v = clamp((v - a) / (b - a)); return v * v * (3 - 2 * v)
def io(v):
    v = clamp(v); return 4 * v ** 3 if v < .5 else 1 - (-2 * v + 2) ** 3 / 2
def out_back(v, s=0.8):
    v = clamp(v) - 1; return 1 + v * v * ((s + 1) * v + s)
def camera(t):
    for a, b in zip(CAM, CAM[1:]):
        if t <= b[0]:
            u = io((t - a[0]) / (b[0] - a[0])) if abs(a[3] - b[3]) > 0.2 or abs(a[1] - b[1]) > 200 else sstep(0, 1, (t - a[0]) / (b[0] - a[0]))
            return [a[i] + (b[i] - a[i]) * u for i in (1, 2, 3)]
    return list(CAM[-1][1:])
def premult(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255
    return np.dstack([rgba[..., :3].astype(np.float32) * a, a])
def comp(cv, sp, x, y, alpha=1.0):
    h, w = sp.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.003: return
    s = sp[y0 - y:y1 - y, x0 - x:x1 - x]; d = cv[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * alpha; d += s[..., :3] * alpha
def draw_sprite(cv, sp, cx, cy, f, alpha=1.0):
    h, w = sp.shape[:2]; dw, dh = w * f, h * f
    x, y = cx - dw / 2, cy - dh / 2; ix, iy = math.floor(x), math.floor(y)
    ow, oh = int(math.ceil(dw)) + 2, int(math.ceil(dh)) + 2
    if ix > W or iy > H or ix + ow < 0 or iy + oh < 0: return
    M = np.float32([[f, 0, x - ix + (f - 1) / 2], [0, f, y - iy + (f - 1) / 2]])
    comp(cv, cv2.warpAffine(sp, M, (ow, oh), flags=cv2.INTER_LINEAR), ix, iy, alpha)

def pill(text, size=23):
    S = 2; font = ImageFont.truetype(FD + "Inter-SemiBold.otf", size * S)
    tw = int(font.getlength(text)); px, py = 16 * S, 9 * S
    w, h = tw + 2 * px + 16 * S, size * S + 2 * py + 6 * S; o = 10 * S
    im = Image.new("RGBA", (w + 2 * o, h + 2 * o), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); ImageDraw.Draw(sh).rounded_rectangle([o, o + 4 * S, o + w, o + h + 4 * S], radius=h // 2, fill=(20, 30, 40, 70))
    from PIL import ImageFilter
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(5 * S))); d = ImageDraw.Draw(im)
    d.rounded_rectangle([o, o, o + w, o + h], radius=h // 2, fill=(255, 255, 255, 245), outline=(40, 170, 190, 255), width=int(1.5 * S))
    d.ellipse([o + 11 * S, o + h / 2 - 5 * S, o + 21 * S, o + h / 2 + 5 * S], fill=(40, 170, 190, 255))
    d.text((o + px + 14 * S, o + py - 1 * S), text, font=font, fill=(28, 38, 48, 255))
    im = im.resize((im.width // S, im.height // S), Image.LANCZOS)
    return premult(np.array(im)[..., [2, 1, 0, 3]])
def text_layer(text, path, size, color, tracking=0):
    font = ImageFont.truetype(path, size); pad = 12
    w = int(sum(font.getlength(c) + tracking for c in text)) + 2 * pad
    im = Image.new("RGBA", (w, int(size * 1.4) + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = pad
    for c in text: d.text((x, pad), c, font=font, fill=color + (255,)); x += font.getlength(c) + tracking
    return premult(np.array(im)[..., [2, 1, 0, 3]])

def load():
    A = {}
    cut = cv2.imread(f"{R}/layers/machine_wide_cut.png", cv2.IMREAD_UNCHANGED); a = cut[..., 3]
    yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
    a[((yy < 396) & (xx > 1204)) | (xx > 1684) | (yy < 380)] = 0                 # building's steel brace
    a[470:566, 1086:1109] = 0
    mach = premult(cut); A["mach"] = mach; PH, PW = mach.shape[:2]
    # soft floor shadow under both units
    sh = np.zeros((PH, PW), np.float32)
    for (cx, cy, rx, ry) in ((690, 935, 250, 34), (400, 905, 170, 16), (1310, 1005, 400, 42)):
        cv2.ellipse(sh, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    A["shadow"] = cv2.GaussianBlur(sh, (0, 0), 26) * 0.42
    cfg = json.load(open(f"{R}/studio_parts.json"))["parts"]
    for i, p in enumerate(cfg):
        poly = p["poly"]
        if not isinstance(poly[0], list): x0, y0, x1, y1 = poly; poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        pts = np.array(poly, np.int32); x0, y0 = pts.min(0) - 6; x1, y1 = pts.max(0) + 6
        m = np.zeros((PH, PW), np.float32); cv2.fillPoly(m, [pts], 1.0, cv2.LINE_AA); m = cv2.GaussianBlur(m, (0, 0), 1.2)
        p["box"] = (x0, y0, x1, y1); p["mask"] = m[y0:y1, x0:x1]
        sp = mach[y0:y1, x0:x1] * p["mask"][..., None]
        # thin teal rim so a lifted part reads as its own object
        al = sp[..., 3]; rim = np.clip(cv2.dilate(al, np.ones((5, 5), np.uint8)) - al, 0, 1)
        p["sp"] = sp; p["rim"] = np.dstack([rim[..., None] * TEAL, rim])
        pad = 30; sa = cv2.GaussianBlur(cv2.copyMakeBorder(al, pad, pad, pad, pad, cv2.BORDER_CONSTANT), (0, 0), 10)
        p["sh"] = np.dstack([np.zeros(sa.shape + (3,), np.float32), sa * 0.5])
        p["home"] = ((x0 + x1) / 2, (y0 + y1) / 2); p["pill"] = pill(p["label"])
        p["t0"] = T_EXP + i * EXP_STAG; p["t1"] = T_RE + (len(cfg) - 1 - i) * RE_STAG
    A["parts"] = cfg
    # studio backdrop
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - 960) / 1250) ** 2 + ((yy - 470) / 900) ** 2)
    v = 240 - 58 * np.clip(r, 0, 1.4) ** 1.6 - 10 * np.clip((yy - 700) / 380, 0, 1)
    A["bg"] = np.dstack([v * 1.0, v * 0.992, v * 0.975]).astype(np.float32)
    A["name"] = text_layer("MARK 8 AUTOMATIC CIGARETTE MAKING MACHINE (MK-8)", FD + "Inter-Bold.otf", 34, (28, 38, 48), 1.5)
    A["brand"] = text_layer("CIVIC TOBACCO MACHINERY  ·  KARACHI - PAKISTAN", FD + "Inter-SemiBold.otf", 21, (150, 120, 20)[::-1], 5)
    return A

def part_e(p, t):
    return out_back((t - p["t0"]) / EXP_DUR) if t < T_RE else 1 - io((t - p["t1"]) / RE_DUR)

def render(t, A):
    cx, cy, z = camera(t); k = S0 * z
    M = np.float32([[k, 0, 960 - cx * k], [0, k, 540 - cy * k]])
    P = lambda x, y: (x * k + M[0, 2], y * k + M[1, 2])
    cv = A["bg"].copy()
    sh = cv2.warpAffine(A["shadow"], M, (W, H), flags=cv2.INTER_LINEAR)
    cv *= 1 - sh[..., None]
    es = [part_e(p, t) for p in A["parts"]]
    # machine body: where a part has lifted out, leave a faint glassy imprint
    base = A["mach"].copy()
    for p, e in zip(A["parts"], es):
        ec = clamp(e)
        if ec <= 0: continue
        x0, y0, x1, y1 = p["box"]; roi = base[y0:y1, x0:x1]; m = p["mask"][..., None] * sstep(0, 0.35, ec)
        al = roi[..., 3:4]; g = roi[..., :3].mean(2, keepdims=True)              # see-through, desaturated copy stays in the seat
        ghost = np.dstack([(g * 0.45 + al * np.array([225, 215, 185], np.float32) * 0.55), roi[..., 3]]) * 0.42
        roi[:] = roi * (1 - m) + ghost * m
    comp(cv, cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR), 0, 0)
    # guide lines from seat to floating part
    lines = np.zeros((H, W), np.float32); pos = []
    for p, e in zip(A["parts"], es):
        hx, hy = p["home"]; x = hx + (p["to"][0] - hx) * e; y = hy + (p["to"][1] - hy) * e
        y += 4 * math.sin(t * 1.3 + hx * 0.01) * clamp(e)
        pos.append((x, y))
        if e > 0.05:
            a, b = P(hx, hy), P(x, y); n = max(int(math.hypot(b[0] - a[0], b[1] - a[1]) / 12), 1)
            for i in range(0, n, 2):
                u0, u1 = i / n, min((i + 1) / n, 1)
                cv2.line(lines, (int(a[0] + (b[0] - a[0]) * u0), int(a[1] + (b[1] - a[1]) * u0)), (int(a[0] + (b[0] - a[0]) * u1), int(a[1] + (b[1] - a[1]) * u1)), 0.55 * clamp(e), 2, cv2.LINE_AA)
            cv2.circle(lines, (int(a[0]), int(a[1])), 5, 0.8 * clamp(e), -1, cv2.LINE_AA)
    cv += lines[..., None] * (TEAL - cv)
    # floating parts
    for p, e, (x, y) in zip(A["parts"], es, pos):
        if e <= 0.001: continue
        ec = clamp(e); f = k * (1 + 0.10 * ec); sx, sy = P(x, y)
        draw_sprite(cv, p["sh"], sx + 14 * ec * z, sy + 26 * ec * z, f, 0.55 * ec)
        draw_sprite(cv, p["rim"], sx, sy, f, 0.85 * sstep(0.1, 0.6, ec))
        draw_sprite(cv, p["sp"], sx, sy, f, 1.0)
    # labels
    for p, e, (x, y) in zip(A["parts"], es, pos):
        a = sstep(p["t0"] + EXP_DUR * 0.55, p["t0"] + EXP_DUR * 0.55 + 0.45, t) * (1 - sstep(T_LAB_OUT, T_LAB_OUT + 0.4, t))
        if a <= 0: continue
        f = k * 1.10; x0, y0, x1, y1 = p["box"]; hw, hh = (x1 - x0) * f / 2, (y1 - y0) * f / 2
        sx, sy = P(x, y); L = p["pill"]; lw, lh = L.shape[1], L.shape[0]; gap = 26
        ax, ay, lx, ly = {"T": (sx, sy - hh + 6, sx, sy - hh - gap - lh / 2 + 10), "B": (sx, sy + hh - 6, sx, sy + hh + gap + lh / 2 - 10),
                          "L": (sx - hw + 6, sy, sx - hw - gap - lw / 2 + 10, sy), "R": (sx + hw - 6, sy, sx + hw + gap + lw / 2 - 10, sy)}[p["side"]]
        if not (-20 < ax < W + 20 and -20 < ay < H + 20 and lw / 2 - 40 < lx < W - lw / 2 + 40 and lh / 2 - 10 < ly < H - lh / 2 + 10): continue
        ln = np.zeros((H, W), np.float32)
        cv2.line(ln, (int(ax), int(ay)), (int(lx), int(ly)), a, 2, cv2.LINE_AA); cv2.circle(ln, (int(ax), int(ay)), 5, a, -1, cv2.LINE_AA)
        cv += ln[..., None] * (TEAL - cv)
        comp(cv, L, int(lx - lw / 2), int(ly - lh / 2 + (1 - a) * 10), a)
    # machine name
    for a, y in ((sstep(0.3, 1.0, t) * (1 - sstep(1.5, 2.1, t)), 968), (sstep(13.3, 14.0, t), 968)):
        if a > 0:
            comp(cv, A["name"], int(960 - A["name"].shape[1] / 2), int(y - A["name"].shape[0] / 2), a)
            comp(cv, A["brand"], int(960 - A["brand"].shape[1] / 2), int(y + 44 - A["brand"].shape[0] / 2), a)
    cv *= sstep(0, 0.4, t) * (1 - sstep(TOTAL - 0.4, TOTAL, t))
    return np.clip(cv, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    A = load()
    if "--still" in sys.argv:
        for v in sys.argv[sys.argv.index("--still") + 1:]: cv2.imwrite(f"{R}/build/st_{float(v):05.2f}.png", render(float(v), A))
        sys.exit()
    out = f"{R}/out/mk8_studio_exploded_view.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = int(TOTAL * FPS)
    for i in range(n): ff.stdin.write(render(i / FPS, A).tobytes())
    ff.stdin.close(); ff.wait(); print("wrote", out)
