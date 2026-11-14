"""Refine rotating-wheel centres in the reference photos by rotational self-similarity."""
import cv2, numpy as np, json
CAND = {
 4: [(925,600,183),(465,933,156),(1815,835,180),(794,845,108),(955,968,58),(1294,339,112),
     (565,650,50),(399,389,135),(574,527,78)],
 3: [(819,321,118),(1033,329,90),(1110,492,72),(1166,404,54),(1521,802,90)],
 2: [(865,686,26),(1010,688,26),(638,646,60),(675,917,34)],
}
def score(g, x, y, r, ang=17):
    M = cv2.getRotationMatrix2D((float(x), float(y)), ang, 1)
    rg = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)
    yy, xx = np.ogrid[:g.shape[0], :g.shape[1]]
    m = ((xx - x) ** 2 + (yy - y) ** 2) < (r * 0.92) ** 2
    return np.abs(rg[m] - g[m]).mean()
out = {}
for i, cs in CAND.items():
    g = cv2.imread(f'ref/{i}.jpg', 0).astype(np.float32)
    res = []
    for (x, y, r) in cs:
        best = (1e9, x, y)
        for step in (6, 2, 1):
            bx, by = best[1], best[2]
            for dx in range(-3 * step, 3 * step + 1, step):
                for dy in range(-3 * step, 3 * step + 1, step):
                    s = score(g, bx + dx, by + dy, r)
                    if s < best[0]: best = (s, bx + dx, by + dy)
        print(i, (x, y, r), '->', best[1:], round(best[0], 1))
        res.append([best[1], best[2], r])
    out[i] = res
json.dump(out, open('wheels.json', 'w'))
