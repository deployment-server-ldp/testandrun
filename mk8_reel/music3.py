"""Original track for render3 timeline (120 BPM). Kick/snare/hats, pulse bass, pads, impacts on
section changes, typing ticks, label pops. No whooshes."""
import numpy as np, wave, os
HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000; DUR = 25.0; N = int(DUR * SR); t = np.arange(N) / SR
L = np.zeros(N); Rr = np.zeros(N); rng = np.random.default_rng(3)
SECT = [1.0, 6.0, 11.5, 15.5, 20.5]
def add(sig, t0, g=1.0, pan=0.0):
    i0 = int(t0 * SR); n = min(len(sig), N - i0)
    if n <= 0 or i0 < 0: return
    L[i0:i0 + n] += sig[:n] * g * (1 - max(0, pan)); Rr[i0:i0 + n] += sig[:n] * g * (1 + min(0, pan))
def env(n, d): return np.exp(-np.arange(n) / SR * d)
def kick():
    n = int(.45 * SR); tt = np.arange(n) / SR; f = 45 + 110 * np.exp(-tt * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 7)
def snare():
    n = int(.3 * SR); x = np.convolve(rng.normal(0, 1, n), [1, -.9], 'same')
    return .6 * x * env(n, 18) + .4 * np.sin(2 * np.pi * 190 * np.arange(n) / SR) * env(n, 25)
def hat():
    n = int(.06 * SR); return np.convolve(rng.normal(0, 1, n), [1, -1], 'same') * env(n, 70)
def boom():
    n = int(2.5 * SR); tt = np.arange(n) / SR; f = 32 + 60 * np.exp(-tt * 6)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 1.6)
def tick(fr=3200, d=150):
    n = int(.03 * SR); return np.sin(2 * np.pi * fr * np.arange(n) / SR) * env(n, d)
roots = [55.0, 43.65, 65.41, 49.0]
for b in range(13):
    t0 = 1.0 + b * 2.0 - 2.0 if b == 0 else 1.0 + (b - 1) * 2.0
    if b == 0: t0 = 0.0
    r = roots[b % 4]; n = int(2.0 * SR); tt = np.arange(n) / SR
    e = np.minimum(1, tt / .3) * np.minimum(1, (2 - tt) / .3)
    pad = sum(np.sin(2 * np.pi * f * tt + p) for f, p in ((r * 4, 0), (r * 4 * 1.19, 1), (r * 6, 2), (r * 8.01, .5)))
    add(pad * e, t0, .035, -.3); add(pad * e, t0 + .01, .035, .3)
    if 1.0 <= t0 < 23:
        for k in range(8):
            m = int(.22 * SR); tt2 = np.arange(m) / SR
            s = sum(np.sin(2 * np.pi * r * 2 * h * tt2) / h for h in range(1, 9)) * env(m, 9)
            add(np.tanh(s * 1.5), t0 + k * .25, .07)
x = 1.0
while x < 23.0:
    beat = int(round((x - 1.0) / .5)); brk = 11.5 <= x < 13.5
    if not brk or beat % 4 == 0: add(kick(), x, .75)
    if beat % 2 == 1 and not brk: add(snare(), x, .32, .1)
    if not brk:
        add(hat(), x, .07, -.4); add(hat(), x + .25, .12, -.4)
    x += .5
for s in SECT: add(boom(), s, .55)
add(boom(), 23.0, .5)
n = int(2 * SR); tt = np.arange(n) / SR
add(sum(np.sin(2 * np.pi * f * tt) for f in (220, 261.6, 329.6, 440)) * env(n, 1.5), 23.0, .05)
# typing ticks for section titles
titles = {1.0: 'FEEDING & CONTROL', 6.0: 'DRUMS & ROLLERS', 11.5: 'TRAY & GUARDS', 15.5: 'DRIVE & CONTROL'}
for t0, ttl in titles.items():
    for c in range(len(ttl)): add(tick(2600 + 400 * (c % 2)), t0 + .1 + .7 * c / len(ttl), .05, .2)
for c in range(30): add(tick(2800), 20.6 + .9 * c / 30, .05, .2)
# label pops (shots start + 0.5 + j*0.45)
shots = [(1.0, 4), (6.0, 3), (8.6, 5), (11.5, 3), (15.5, 5)]
for s0, k in shots:
    for j in range(k): add(tick(1500, 80), s0 + .5 + j * .45 + .15, .1, -.2)
fade = np.clip((DUR - t) / .5, 0, 1)
st = np.tanh(np.stack([L * fade, Rr * fade], 1) * 1.3); st /= np.abs(st).max() * 1.05
with wave.open(os.path.join(HERE, 'music3.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype(np.int16).tobytes())
print('ok')
