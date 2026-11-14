"""Cut every component group out of the infographic into its own RGBA layer (layers/parts/<id>.png, 2x)."""
import json, os, cv2, numpy as np
from scipy import ndimage as ndi
R = os.path.join(os.path.dirname(__file__), "..")
UP = 2
img = cv2.imread(f"{R}/source/infographic.webp")
big = cv2.resize(img, None, fx=UP, fy=UP, interpolation=cv2.INTER_LANCZOS4)
blur = cv2.GaussianBlur(big, (0, 0), 1.6)
big = cv2.addWeighted(big, 1.45, blur, -0.45, 0)
os.makedirs(f"{R}/layers/parts", exist_ok=True)
parts = json.load(open(f"{R}/parts.json"))["parts"]
tiles = []
for p in parts:
    x0, y0, x1, y1 = [v * UP for v in p["src"]]
    c = big[y0:y1, x0:x1].copy()
    b, g, r = [c[..., i].astype(np.float32) for i in range(3)]
    warm = np.maximum(r, g * 0.92)
    sat = (c.max(2).astype(np.float32) - c.min(2))
    m = (warm > 78) | ((b > 120) & (sat > 90) & (g < 0.8 * b))   # body colours, or vivid blue objects
    for e in p.get("erase", []):
        ex0, ey0, ex1, ey1 = [v * UP for v in e]
        m[max(ey0 - y0, 0):max(ey1 - y0, 0), max(ex0 - x0, 0):max(ex1 - x0, 0)] = False
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 60 * UP:
            m[lab == i] = 0
    # grow the bright mask so dark/black components next to it stay in the layer
    # (the surrounding navy is invisible on the film's navy background)
    k = p.get("grow", 29)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 5)
    a = np.clip((a - 0.3) / 0.6, 0, 1)
    # tone down leftover cyan callout glow outside the bodies
    cy = ((b > 140) & (r < 0.62 * b) & (sat > 70)).astype(np.float32)
    if not p.get("keep_blue"):
        cy = cv2.GaussianBlur(cv2.dilate(cy, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        dark = cv2.GaussianBlur(cv2.erode(c, np.ones((9, 9), np.uint8)), (0, 0), 3)
        c = (c * (1 - cy[..., None]) + dark * cy[..., None]).astype(np.uint8)
    f = 10  # feather to the crop edge
    ramp = lambda n_: np.clip(np.minimum(np.arange(n_), n_ - 1 - np.arange(n_)) / f, 0, 1)
    a *= ramp(a.shape[0])[:, None] * ramp(a.shape[1])[None, :]
    out = np.dstack([c, (a * 255).astype(np.uint8)])
    cv2.imwrite(f"{R}/layers/parts/{p['id']}.png", out)
    bg = np.full_like(c, (40, 16, 6)); bg[::16] = (90, 40, 10)
    tiles.append((p["id"], (c * a[..., None] + bg * (1 - a[..., None])).astype(np.uint8)))
# contact sheet
W = 1900; x = y = rowh = 0; sheet = np.zeros((2600, W, 3), np.uint8)
for name, t in tiles:
    h, w = t.shape[:2]
    if x + w > W: x = 0; y += rowh + 30; rowh = 0
    sheet[y + 24:y + 24 + h, x:x + w] = t
    cv2.putText(sheet, name, (x + 2, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    x += w + 10; rowh = max(rowh, h + 24)
cv2.imwrite(f"{R}/build/sheet.png", sheet[:y + rowh + 10])
print(y + rowh)
