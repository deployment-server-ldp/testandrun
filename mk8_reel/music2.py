"""Original cinematic-industrial track, 120 BPM, synced to render2 timeline. No whooshes."""
import numpy as np, wave, sys, os
sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import render2 as R
SR = 48000; N = int(R.DUR * SR); t = np.arange(N) / SR
L = np.zeros(N); Rr = np.zeros(N)
BEAT = 0.5
rng = np.random.default_rng(7)
def add(sig, t0, gain=1.0, pan=0.0):
    i0 = int(t0 * SR); n = min(len(sig), N - i0)
    if n <= 0 or i0 < 0: return
    L[i0:i0 + n] += sig[:n] * gain * (1 - max(0, pan)); Rr[i0:i0 + n] += sig[:n] * gain * (1 + min(0, pan))
def kick():
    tt = np.arange(int(0.45 * SR)) / SR
    f = 45 + 110 * np.exp(-tt * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 7) + 0.3 * np.exp(-tt * 200) * rng.normal(0, 1, len(tt))
def snare():
    tt = np.arange(int(0.3 * SR)) / SR
    n = rng.normal(0, 1, len(tt)); n = np.convolve(n, [1, -0.9], 'same')
    return 0.6 * n * np.exp(-tt * 18) + 0.4 * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 25)
def hat():
    tt = np.arange(int(0.06 * SR)) / SR
    n = np.convolve(rng.normal(0, 1, len(tt)), [1, -1], 'same')
    return n * np.exp(-tt * 70)
def boom():
    tt = np.arange(int(2.5 * SR)) / SR
    f = 32 + 60 * np.exp(-tt * 6)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.6)
    n = rng.normal(0, 1, len(tt)); n = np.cumsum(n); n -= np.convolve(n, np.ones(200) / 200, 'same')
    return s + 0.04 * n * np.exp(-tt * 4)
def click():
    tt = np.arange(int(0.05 * SR)) / SR
    return np.sin(2 * np.pi * 2600 * tt) * np.exp(-tt * 120) + 0.5 * np.sin(2 * np.pi * 900 * tt) * np.exp(-tt * 90)
def saw(f, dur):
    tt = np.arange(int(dur * SR)) / SR
    s = sum(np.sin(2 * np.pi * f * h * tt) / h for h in range(1, 9))
    return s
# chord roots (A minor: Am F C G), one chord per bar (2 s)
roots = [55.0, 43.65, 65.41, 49.0]
for b in range(int(R.DUR / 2)):
    t0 = b * 2.0
    r = roots[b % 4]
    third = r * (1.2 if b % 4 in (0,) else 1.26)
    # pad
    tt = np.arange(int(2.0 * SR)) / SR
    env = np.minimum(1, tt / 0.4) * np.minimum(1, (2.0 - tt) / 0.4)
    pad = sum(np.sin(2 * np.pi * fr * tt + ph) for fr, ph in ((r * 4, 0), (third * 4, 1), (r * 6, 2), (r * 8.01, .5)))
    lev = 0.05 if t0 < 4 else 0.035
    add(pad * env, t0, lev, -0.3); add(pad * env, t0 + 0.01, lev, 0.3)
    # pulse bass on 8ths after intro
    if t0 >= 4 and t0 < 34:
        for e in range(8):
            s = saw(r * 2, 0.22) * np.exp(-np.arange(int(0.22 * SR)) / SR * 9)
            add(np.tanh(s * 1.5), t0 + e * 0.25, 0.07)
# drums
intro_kicks = [0.0, 2.0]
for tk in intro_kicks: add(kick(), tk, 0.5)
x = 4.0
while x < 34.0:
    sparse = (24.0 <= x < 26.0)  # breakdown on the tray close-up
    beat = int(round((x - 4.0) / BEAT))
    if not sparse or beat % 4 == 0: add(kick(), x, 0.75)
    if beat % 2 == 1 and x >= 6.0 and not sparse: add(snare(), x, 0.35, 0.1)
    for hh in range(2):
        if x >= 6.0 and not sparse: add(hat(), x + hh * 0.25, 0.12 if hh else 0.07, -0.4)
    x += BEAT
# impacts on section changes
for tb in (4.0, 6.0, 16.0, 26.0, 32.0):
    add(boom(), tb, 0.6)
add(boom(), 34.0, 0.5)
# final chord ring-out
tt = np.arange(int(2.0 * SR)) / SR
add(sum(np.sin(2 * np.pi * f * tt) for f in (220, 261.6, 329.6, 440)) * np.exp(-tt * 1.5), 34.0, 0.05)
# label ticks and lock clicks
for p in R.P:
    add(click(), R.T_LAB0 + p['idx'] * 0.3 + 0.25, 0.06, 0.3)
    add(click(), R.asm_end(p), 0.14, -0.2)
for c in R.CUTS:
    tt = np.arange(int(0.12 * SR)) / SR
    add(rng.normal(0, 1, len(tt)) * np.exp(-tt * 40) * (np.sin(2 * np.pi * 60 * tt) > 0), c - 0.05, 0.08)
fade = np.clip((R.DUR - t) / 0.5, 0, 1)
st = np.stack([L * fade, Rr * fade], 1)
st = np.tanh(st * 1.3)
st /= np.abs(st).max() * 1.05
with wave.open(os.path.join(HERE, 'music2.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype(np.int16).tobytes())
print('ok')
