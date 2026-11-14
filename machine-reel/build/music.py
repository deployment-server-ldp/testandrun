#!/usr/bin/env python3
"""Soundtrack: the music of source/mk8_reel_v3.mp4 (120 BPM, chords Fm-Cm-G-Am) re-cut to the film,
plus original synthesized layers in the same key (hats, claps, rolls, risers, crashes, bass, arps, lead)
arranged differently for every scene. Finally sped up one semitone (x1.059) together with the picture.
usage: music.py <out.wav>"""
import os, subprocess, sys, wave, numpy as np
from scipy import signal
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SR = 48000
SPEED_UP, SPEED_DN = 1059, 1000                # must match SPEED in render.py
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{R}/source/mk8_reel_v3.mp4", "-vn", "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"],
                     capture_output=True, check=True).stdout
src = np.frombuffer(raw, "<i2").reshape(-1, 2).astype(np.float64) / 32768
# (film start, source start, length) - all on the 0.5 s beat grid, film grid offset 4.1 s
EDL = [(4.1, 1.0, 4.0), (8.1, 6.0, 7.0), (15.1, 13.0, 2.5), (17.6, 18.0, 2.5), (20.1, 1.0, 19.5), (39.6, 20.5, 4.5)]
TOTAL = 44.1; N = int(TOTAL * SR)
PRE, XF = 0.015, 0.012
base = np.zeros((N + SR, 2))
def place(seg, at, fi, fo):
    seg = seg.copy(); a, b = int(fi * SR), int(fo * SR)
    if a: seg[:a] *= np.linspace(0, 1, a)[:, None]
    if b: seg[-b:] *= np.linspace(1, 0, b)[:, None]
    i = int(round(at * SR)); base[i:i + len(seg)] += seg
for k, (ft, st, ln) in enumerate(EDL):
    last = k == len(EDL) - 1
    s0 = int(round((st - PRE) * SR)); s1 = int(round((st + ln - PRE + (0 if last else XF)) * SR))
    place(src[s0:min(s1, len(src))], ft - PRE, XF, 0 if last else XF)
dr = src[int(0.30 * SR):int(0.96 * SR)]; n = len(dr); xf = int(0.25 * SR)
w = np.ones(n); w[:xf] = np.sin(np.linspace(0, np.pi / 2, xf)) ** 2; w[-xf:] = np.cos(np.linspace(0, np.pi / 2, xf)) ** 2
pos = 0
while pos < int(4.1 * SR):
    e = min(pos + n, int(4.1 * SR)); base[pos:e] += (dr * w[:, None])[:e - pos]; pos += n - xf
base[:int(4.1 * SR)] *= np.clip(np.arange(int(4.1 * SR)) / (1.2 * SR), 0, 1)[:, None]
tail = src[int(23.05 * SR):int(24.95 * SR)][::-1].copy(); tail *= (np.linspace(0, 1, len(tail)) ** 1.5)[:, None]
i = int((4.1 - PRE) * SR) - len(tail); base[i:i + len(tail)] += tail * 0.8
base = base[:N]

# ------------------------------------------------------------------ added layers
rng = np.random.default_rng(5)
L = np.zeros(N + 4 * SR); Rr = np.zeros(N + 4 * SR)          # drums / fx (not ducked)
ML = np.zeros(N + 4 * SR); MR = np.zeros(N + 4 * SR)         # musical layers (ducked under the kick)
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
def put(bufs, x, at, g=1.0, pan=0.0):
    i = int(round(at * SR))
    if i < 0: x = x[-i:]; i = 0
    bufs[0][i:i + len(x)] += x * g * (1 - max(pan, 0)); bufs[1][i:i + len(x)] += x * g * (1 + min(pan, 0))
D, M = (L, Rr), (ML, MR)
def filt(x, lo=None, hi=None, order=2):
    if lo and hi: sos = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], "band", output="sos")
    elif lo: sos = signal.butter(order, lo / (SR / 2), "high", output="sos")
    else: sos = signal.butter(order, hi / (SR / 2), output="sos")
    return signal.sosfilt(sos, x)
T_ = lambda d: np.arange(int(d * SR)) / SR
def src_time(ft):
    for f0, s0, ln in EDL:
        if f0 <= ft < f0 + ln: return s0 + ft - f0
    return 0.0
CHORDS = [(41, [53, 56, 60]), (36, [48, 51, 55]), (43, [55, 62, 67]), (45, [57, 60, 64])]   # Fm, Cm, G(5), Am
def chord(ft):
    s = src_time(ft)
    return CHORDS[3] if s < 1.0 else CHORDS[int((s - 1.0) // 2) % 4]

def hat(d=0.045, f=7500):
    t = T_(d); return filt(rng.normal(0, 1, len(t)), lo=f) * np.exp(-t * (3.2 / d))
def ohat(): t = T_(0.22); return filt(rng.normal(0, 1, len(t)), lo=6000) * np.exp(-t * 14)
def clap():
    t = T_(0.25); x = np.zeros(len(t))
    for o in (0, 0.011, 0.023): 
        k = int(o * SR); x[k:] += rng.normal(0, 1, len(t) - k) * np.exp(-T_(0.25)[:len(t) - k] * (60 if o < 0.02 else 20))
    return filt(x, lo=800, hi=3800)
def snare():
    t = T_(0.16); return (filt(rng.normal(0, 1, len(t)), lo=1500) * np.exp(-t * 28) + 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 40))
def tom(f):
    t = T_(0.28); return np.sin(2 * np.pi * np.cumsum(f * (1 + 0.5 * np.exp(-t * 30))) / SR) * np.exp(-t * 12)
def crash():
    t = T_(2.4); return filt(rng.normal(0, 1, len(t)), lo=3500) * (np.exp(-t * 2.2) + 0.6 * np.exp(-t * 14))
def boom():
    t = T_(2.0); f = 32 + 55 * np.exp(-t * 10)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.8) + filt(rng.normal(0, 1, len(t)), hi=700) * np.exp(-t * 9) * 0.6
def riser(d):
    n = int(d * SR); x = rng.normal(0, 1, n); out = np.zeros(n); B = 2400
    for i in range(0, n, B):
        fc = 400 * (11000 / 400) ** (i / n)
        seg = filt(x[i:i + B + 600], lo=fc * 0.75, hi=min(fc * 1.35, 20000))
        out[i:i + B] = seg[600:600 + B][:len(out[i:i + B])] if len(seg) >= B + 600 else seg[:len(out[i:i + B])]
    t = T_(d)[:n]
    tone = np.sin(2 * np.pi * np.cumsum(220 * 2 ** (t / d * 2)) / SR) * 0.25
    return (out + tone) * (t / d) ** 2.2
def roll(t0, t1, g=0.16):                                   # accelerating snare roll into t1
    t = t0; d = t1 - t0
    while t < t1 - 0.01:
        u = (t - t0) / d
        put(D, SN, t, g * (0.35 + 0.65 * u), 0.15 * np.sin(t * 40))
        t += 0.125 if u < 0.5 else 0.0625
def bassnote(m, d=0.115):
    t = T_(d); f = hz(m); x = np.zeros(len(t))
    for h in range(1, 9): x += np.sin(2 * np.pi * f * h * t) / h * np.exp(-t * (4 + 5 * h))
    return x * np.minimum(t / .004, 1) * np.minimum((d - t) / .01, 1)
def pluck(m, d=0.42, bright=1.0):
    t = T_(d); f = hz(m); x = np.zeros(len(t))
    for h in range(1, 7): x += np.sin(2 * np.pi * f * h * t) / h ** (1.5 / bright) * np.exp(-t * (8 + 5 * h))
    return x * np.minimum(t / .003, 1)
def lead(m, d=0.3):
    t = T_(d); f = hz(m) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t)); ph = 2 * np.pi * np.cumsum(f) / SR; x = np.zeros(len(t))
    for h in range(1, 10): x += np.sin(ph * h) / h
    return filt(x, hi=3200) * np.minimum(t / .006, 1) * np.exp(-t * 6)
def stab(notes, d=0.16):
    t = T_(d); x = np.zeros(len(t))
    for m in notes:
        for det in (-0.005, 0.005):
            for h in range(1, 7): x += np.sin(2 * np.pi * hz(m) * (1 + det) * h * t + h) / h
    return filt(x, hi=2600) * np.minimum(t / .004, 1) * np.exp(-t * 16)
def padchord(notes, d):
    t = T_(d); x = np.zeros(len(t))
    for m in notes:
        for det in (-0.004, 0, 0.004):
            for h in range(1, 6): x += np.sin(2 * np.pi * hz(m) * (1 + det) * h * t + rng.random() * 6) / h ** 1.7
    return filt(x, hi=1800) * np.minimum(t / 0.5, 1) * np.minimum((d - t) / 1.5, 1)
H, HO, OH, CL, SN, CR, BM = hat(), hat(0.03, 9000), ohat(), clap(), snare(), crash(), boom()

G0 = 4.1                                                   # beat grid origin
def steps(a, b):                                           # 16th-note steps inside [a,b)
    k0 = int(np.ceil((a - G0) / 0.125 - 1e-6)); k1 = int(np.ceil((b - G0) / 0.125 - 1e-6))
    return [(k, G0 + k * 0.125) for k in range(k0, k1)]
def drums(a, b, hats=1.0, claps=1.0, open_=0.0, shaker=0.0):
    for k, t in steps(a, b):
        q = k % 4
        if hats: put(D, H if q % 2 == 0 else HO, t, 0.075 * hats * (1.0, 0.45, 0.8, 0.5)[q], 0.25)
        if open_ and q == 2: put(D, OH, t, 0.07 * open_, -0.2)
        if claps and k % 8 == 4: put(D, CL, t, 0.16 * claps)
        if claps and k % 32 == 30: put(D, SN, t, 0.07 * claps, 0.2)
def bassline(a, b, g=0.15, octave=True):
    for k, t in steps(a, b):
        q = k % 4
        if q == 0: continue                                # leave the down-beat to the track's own kick
        root = chord(t)[0]
        put(M, bassnote(root + (12 if octave and q == 3 else 0)), t, g * (1.0 if q == 2 else 0.8))
def arp(a, b, g=0.05, oct_=12, every=1, bright=1.0):
    seq = [0, 1, 2, 3, 2, 1, 2, 3, 0, 2, 1, 3, 2, 3, 1, 2]
    for k, t in steps(a, b):
        if k % every: continue
        tones = chord(t)[1]; i = seq[(k // every) % 16]
        m = (tones[i] if i < 3 else tones[0] + 12) + oct_
        put((arpL, arpR), pluck(m, bright=bright), t, g * (1.0 if k % 4 == 0 else 0.75), 0.6 if (k // every) % 2 else -0.6)
def leadline(a, b, g=0.05, var=0):
    pat = [[2, None, 1, 0, None, 1, 2, 3], [3, 2, None, 2, 1, None, 0, 1]][var]
    for k, t in steps(a, b):
        if k % 2: continue
        i = pat[(k // 2) % 8]
        if i is None: continue
        tones = chord(t)[1]; m = (tones[i] if i < 3 else tones[0] + 12) + 12
        put((arpL, arpR), lead(m), t, g, 0.0)
def stabs(a, b, g=0.035):
    for k, t in steps(a, b):
        if k % 4 == 2: put(M, stab(chord(t)[1]), t, g, 0.3 if k % 8 == 2 else -0.3)
def hit(t, g=1.0, cr=True):
    put(D, BM, t, 0.42 * g)
    if cr: put(D, CR, t, 0.13 * g, 0.1)
def build(t1, d, g=1.0, r=True):
    put(D, riser(d), t1 - d, 0.12 * g)
    put(D, CR[::-1] * np.linspace(0, 1, len(CR)) ** 3, t1 - len(CR) / SR, 0.07 * g, -0.1)
    if r: roll(t1 - min(d, 1.0), t1, 0.15 * g)
arpL = np.zeros(N + 4 * SR); arpR = np.zeros(N + 4 * SR)

# --- intro 0 - 4.1: pulse, teaser arp, build into the explosion
for t in (0.6, 1.6, 2.6): put(D, BM[:int(0.7 * SR)] * np.exp(-T_(0.7) * 4), t, 0.22)
for j, t in enumerate(np.arange(1.1, 4.1, 0.25)):
    tones = CHORDS[3][1]; put((arpL, arpR), pluck(tones[j % 3] + (12 if j % 6 > 2 else 0), bright=0.7 + j * 0.04), t, 0.012 + 0.004 * j, 0.5 if j % 2 else -0.5)
build(4.1, 2.0)
# --- explosion 4.1 - 8.1
hit(4.1, 1.2); drums(4.1, 8.1, hats=1.0, claps=1.0); bassline(4.1, 8.1); stabs(6.1, 8.1, 0.03)
for i, f in enumerate((180, 150, 120, 95)): put(D, tom(f), 7.6 + i * 0.125, 0.22, -0.4 + 0.27 * i)
# --- labels 8.1 - 13.6: arp joins, open hats
hit(8.1, 0.9); drums(8.1, 13.6, hats=1.0, claps=1.0, open_=1.0); bassline(8.1, 13.6); arp(8.1, 13.6, 0.05)
leadline(10.1, 13.6, 0.04, 0)
# --- break 13.6 - 15.1 (labels leave) -> build
put(D, CR, 13.6, 0.08); build(15.1, 1.5)
arp(13.6, 15.1, 0.035, 24, every=2)
# --- reassembly 15.1 - 20.1: full, arp an octave up, lead answers, long build to the tour
hit(15.1, 1.0); drums(15.1, 20.1, hats=1.0, claps=1.1, open_=1.0); bassline(15.1, 20.1, 0.16)
arp(15.1, 20.1, 0.045, 24); leadline(15.1, 19.1, 0.05, 1); stabs(15.1, 20.1, 0.03)
build(20.1, 2.0)
# --- tour 1: 20.1 - 25.1  drums + bass + chord stabs (no arp: new colour)
hit(20.1, 1.2); drums(20.1, 25.1, hats=1.0, claps=1.0); bassline(20.1, 25.1, 0.16); stabs(20.1, 25.1, 0.045)
leadline(22.1, 25.1, 0.04, 0)
for i, f in enumerate((200, 160, 130, 100)): put(D, tom(f), 24.6 + i * 0.125, 0.22, 0.4 - 0.27 * i)
# --- tour 2: 25.1 - 30.6  arp + open hats, brighter
hit(25.1, 0.9); drums(25.1, 30.6, hats=1.0, claps=1.0, open_=1.2); bassline(25.1, 30.6, 0.15); arp(25.1, 30.6, 0.05, 12, bright=1.3)
leadline(27.1, 30.6, 0.045, 1)
# --- break 30.6 - 32.1
put(D, CR, 30.6, 0.08); arp(30.6, 32.1, 0.035, 24, every=2); build(32.1, 1.5)
# --- tour 3: 32.1 - 34.6  breakdown: toms + sparse high arp, no hats / bass
hit(32.1, 0.8)
for k, t in steps(32.1, 34.6):
    if k % 8 in (3, 6): put(D, tom(110 if k % 8 == 3 else 85), t, 0.2, 0.3 if k % 8 == 3 else -0.3)
    if k % 8 == 4: put(D, CL, t, 0.15)
arp(32.1, 34.6, 0.05, 24, every=2); stabs(32.1, 34.6, 0.04)
roll(34.1, 34.6, 0.14); put(D, riser(1.0), 33.6, 0.1)
# --- tour 4: 34.6 - 39.6  everything
hit(34.6, 1.1); drums(34.6, 39.6, hats=1.1, claps=1.1, open_=1.2); bassline(34.6, 39.6, 0.17)
arp(34.6, 39.6, 0.045, 24, bright=1.3); leadline(34.6, 38.6, 0.055, 0); stabs(34.6, 39.6, 0.035)
build(39.6, 2.0)
# --- final 39.6 - 44.1
hit(39.6, 1.3); drums(39.6, 42.1, hats=0.8, claps=0.9); bassline(39.6, 42.1, 0.13); arp(39.6, 42.6, 0.04, 12)
put(M, padchord([45, 57, 60, 64, 69, 76], 4.4), 39.7, 0.03); put(D, CR, 42.1, 0.07)

# echo on arp / lead, then into the ducked bus
d = int(0.375 * SR)
for buf in (arpL, arpR):
    e = buf.copy()
    for k in range(1, 4): e[k * d:] += buf[:-k * d] * 0.4 ** k
    buf[:] = e
ML += arpL + 0.25 * arpR; MR += arpR + 0.25 * arpL
# sidechain: duck the musical layers on every beat of the track's kick
t = np.arange(len(ML)) / SR
ph = np.mod(t - G0, 0.5)
duck = np.where((t >= G0) & (t < 42.2), 0.35 + 0.65 * np.clip(ph / 0.22, 0, 1) ** 1.5, 1.0)
for a, b in ((13.6, 15.1), (30.6, 32.1)): duck[(t >= a) & (t < b)] = 1.0
ML *= duck; MR *= duck
# short room on drums / music
def verb(x, seed, d=1.3, k=3.5):
    r = np.random.default_rng(seed); tt = T_(d); ir = filt(r.normal(0, 1, len(tt)), hi=6000) * np.exp(-tt * k); ir[:int(.01 * SR)] = 0
    return signal.fftconvolve(x, ir / np.sqrt((ir ** 2).sum()))[:len(x)]
lay = np.stack([L + ML + 0.16 * verb(L * 0.6 + ML, 1), Rr + MR + 0.16 * verb(Rr * 0.6 + MR, 2)], 1)[:N]
LAYER_GAIN = float(os.environ.get("LAYER_GAIN", "1.9"))
seg = slice(int(21 * SR), int(30 * SR))
print("rms base %.3f layers %.3f" % (np.sqrt((base[seg] ** 2).mean()) * 0.7, np.sqrt((lay[seg] ** 2).mean()) * LAYER_GAIN))
out = base * 0.7 + lay * LAYER_GAIN
out[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))[:, None]
out = filt(out.T, lo=25).T
pk = np.percentile(np.abs(out), 99.95)
out = np.tanh(out / pk * 1.7) * 0.86
out = signal.resample_poly(out, SPEED_DN, SPEED_UP, axis=0)            # faster + one semitone up
with wave.open(sys.argv[1], "wb") as wv:
    wv.setnchannels(2); wv.setsampwidth(2); wv.setframerate(SR); wv.writeframes((np.clip(out, -1, 1) * 32767).astype("<i2").tobytes())
