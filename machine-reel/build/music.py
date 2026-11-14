#!/usr/bin/env python3
"""Original synthesized score for the reel (no samples, no third-party music).
usage: music.py <seconds> <out.wav>   -- 120 BPM, D minor, cues locked to the scene changes."""
import sys, wave, numpy as np
from scipy import signal
SR = 48000
DUR = float(sys.argv[1]); OUT = sys.argv[2]
N = int(DUR * SR); t = np.arange(N) / SR
rng = np.random.default_rng(3)
L = np.zeros(N); Rr = np.zeros(N)
TOUR0, TOUR1 = 22.0, DUR - 4.0
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
def add(buf, x, at, g=1.0):
    i = int(at * SR)
    if i >= N or i + len(x) <= 0: return
    j0 = max(-i, 0); i = max(i, 0); n = min(len(x) - j0, N - i)
    buf[i:i + n] += x[j0:j0 + n] * g
def st(x, at, g=1.0, pan=0.0):
    add(L, x, at, g * (1 - max(pan, 0))); add(Rr, x, at, g * (1 + min(pan, 0)))
def lp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc / (SR / 2), output="sos"), x)
def hp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc / (SR / 2), "high", output="sos"), x)

# chords: Dm | Bb | F | C, 4 s each (two bars)
CH = [(38, [62, 65, 69, 74]), (34, [58, 62, 65, 70]), (41, [60, 65, 69, 72]), (36, [60, 64, 67, 72])]
chord_at = lambda s: CH[int(s // 4) % 4]

# --- pad (soft detuned saws, slow envelope)
def pad(notes, d):
    n = int(d * SR); tt = np.arange(n) / SR; x = np.zeros(n)
    for m in notes:
        for det in (-0.004, 0.0, 0.004):
            f = hz(m) * (1 + det)
            for h in range(1, 7):
                x += np.sin(2 * np.pi * f * h * tt + rng.random() * 6.28) / h ** 1.6
    env = np.minimum(tt / 1.2, 1) * np.minimum((d - tt) / 1.4, 1)
    return lp(x, 1500) * env
for s in np.arange(0, DUR, 4):
    root, notes = chord_at(s)
    lvl = 0.022 if s < 4 else 0.016
    p = pad(notes, 5.2)
    st(p, s, lvl, 0.25); st(np.roll(p, 301), s, lvl, -0.25)
    dr = pad([root, root + 12], 5.2)                       # low drone
    st(dr, s, 0.022)

# --- drums / bass / arp, on from the explosion to the end of the close-up tour
def kick():
    tt = np.arange(int(.32 * SR)) / SR
    f = 46 + 95 * np.exp(-tt * 34)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 11)
def hat(d=0.05):
    tt = np.arange(int(d * SR)) / SR
    return hp(rng.normal(0, 1, len(tt)), 7000) * np.exp(-tt * 70)
def clap():
    tt = np.arange(int(.22 * SR)) / SR
    return signal.sosfilt(signal.butter(2, [900 / (SR / 2), 3200 / (SR / 2)], "band", output="sos"), rng.normal(0, 1, len(tt))) * np.exp(-tt * 22)
def bass(m, d=0.42):
    tt = np.arange(int(d * SR)) / SR; f = hz(m)
    return (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(4 * np.pi * f * tt)) * np.minimum(tt / .008, 1) * np.exp(-tt * 5)
def pluck(m, d=0.5):
    tt = np.arange(int(d * SR)) / SR; f = hz(m); x = np.zeros(len(tt))
    for h in range(1, 6): x += np.sin(2 * np.pi * f * h * tt) / h ** 1.3 * np.exp(-tt * (7 + 4 * h))
    return x * np.minimum(tt / .003, 1)
K, HT, CL = kick(), hat(), clap()
arpL = np.zeros(N); arpR = np.zeros(N)
for s in np.arange(4.0, TOUR1 - 0.01, 0.25):               # 16th-note grid at 120 BPM = 0.125; we use 8ths
    root, notes = chord_at(s)
    full = s >= 10.0
    beat = round((s - 4.0) / 0.25)
    brk = (15.5 <= s < 16.0) or (21.5 <= s < 22.0) or (TOUR1 - 0.5 <= s)   # short breaks before scene changes
    if beat % 4 == 0 and not brk: st(K, s, 0.50 if full else 0.36)
    if beat % 4 == 2 and not brk and (s < 10 or True): st(K, s, 0.22 if full else 0.0)
    if full and beat % 8 == 4 and not brk: st(CL, s, 0.11)
    if full and beat % 2 == 1: st(HT, s, 0.05, 0.3)
    if not brk: st(bass(root if beat % 8 < 6 else root + 7), s, 0.20 if full else 0.14)
    if s >= 6.0:
        seq = [0, 1, 2, 3, 2, 1, 3, 2]
        m = notes[seq[beat % 8]] + (12 if (beat // 8) % 2 and full else 0)
        g = 0.045 if full else 0.03
        if TOUR0 <= s: g = 0.052
        add(arpL if beat % 2 == 0 else arpR, pluck(m), s, g)
        if full and beat % 2 == 1: add(arpR if beat % 2 == 0 else arpL, pluck(m + 12, 0.3), s + 0.125, g * 0.45)
# dotted-eighth echo on the arp
d = int(0.375 * SR)
for buf in (arpL, arpR):
    e = buf.copy()
    for k in range(1, 4): e[k * d:] += buf[:-k * d] * 0.42 ** k
    buf[:] = e
L += arpL + 0.3 * arpR; Rr += arpR + 0.3 * arpL

# --- cinematic hits and risers on the scene changes
def boom(d=2.2):
    tt = np.arange(int(d * SR)) / SR
    f = 34 + 40 * np.exp(-tt * 9)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 2.6) + lp(rng.normal(0, 1, len(tt)), 900) * np.exp(-tt * 7) * 0.5
def riser(d=1.6):
    n = int(d * SR); tt = np.arange(n) / SR; x = rng.normal(0, 1, n); out = np.zeros(n); B = 2048
    for i in range(0, n, B):                                    # band-pass sweeping upward
        fc = 300 * (9000 / 300) ** (i / n)
        sos = signal.butter(2, [fc * 0.7 / (SR / 2), min(fc * 1.4, 20000) / (SR / 2)], "band", output="sos")
        seg = signal.sosfilt(sos, x[max(i - B, 0):i + B])[-min(B, n - i) - 0:][:B] if i else signal.sosfilt(sos, x[:B])
        out[i:i + B] = seg[:len(out[i:i + B])]
    return out * (tt / d) ** 2.5
def sweep(d=0.5):
    n = int(d * SR); tt = np.arange(n) / SR
    return hp(rng.normal(0, 1, n), 2500) * np.sin(np.pi * tt / d) ** 2
BM, RS, SW = boom(), riser(), sweep()
for c, g in ((0.35, 0.45), (4.1, 0.75), (10.0, 0.4), (16.2, 0.5), (20.75, 0.6), (TOUR0, 0.6), (TOUR1, 0.85)):
    st(BM, c, g)
for c in (4.1, 16.2, TOUR0, TOUR1): st(RS, c - 1.6, 0.11)
for c in (10.0, 20.4, 26.92, 32.42, 36.42): st(SW, c - 0.25, 0.07, 0.2)
for i in range(19):                                              # soft ticks as each label appears
    tt = np.arange(int(.06 * SR)) / SR
    st(np.sin(2 * np.pi * (1760 if i % 2 else 2093) * tt) * np.exp(-tt * 70), 10.1 + i * 0.25, 0.035, 0.5 if i % 2 else -0.5)
# final chord swell (D minor add9) under the closing title
fin = pad([50, 57, 62, 65, 69, 76], 4.0)
st(fin, TOUR1, 0.04)

# --- reverb + master
def reverb(x, seed):
    r = np.random.default_rng(seed); n = int(2.2 * SR); tt = np.arange(n) / SR
    ir = lp(r.normal(0, 1, n), 5000) * np.exp(-tt * 2.6); ir[:int(.012 * SR)] = 0
    return signal.fftconvolve(x, ir / np.sqrt((ir ** 2).sum()))[:N]
wl, wr = reverb(hp(L, 180), 11), reverb(hp(Rr, 180), 12)
L = L + 0.30 * wl; Rr = Rr + 0.30 * wr
env = np.minimum(t / 0.4, 1) * np.clip((DUR - t) / 1.6, 0, 1) ** 1.5
m = np.stack([L, Rr], 1) * env[:, None]
m = hp(m.T, 28).T
m = np.tanh(m / np.percentile(np.abs(m), 99.9) * 1.1) * 0.89
with wave.open(OUT, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((m * 32767).astype("<i2").tobytes())
