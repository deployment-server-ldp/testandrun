"""Synthesised cinematic sound design for the 15 s reel (48 kHz stereo WAV)."""
import numpy as np, wave
from scipy.signal import butter, sosfilt

SR, DUR = 48000, 15.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)
L = np.zeros(N); R = np.zeros(N)

def env(t0, a, d, length=None):
    x = t - t0
    e = np.where(x < 0, 0, np.where(x < a, x / max(a, 1e-4), np.exp(-(x - a) / d)))
    if length: e *= (x < length)
    return e

def band(sig, lo, hi):
    return sosfilt(butter(2, [lo, hi], "bandpass", fs=SR, output="sos"), sig)

def lp(sig, f): return sosfilt(butter(2, f, "lowpass", fs=SR, output="sos"), sig)

def add(sig, g=1.0, pan=0.0):
    global L, R
    L += sig * g * np.sqrt((1 - pan) / 2); R += sig * g * np.sqrt((1 + pan) / 2)

noise = rng.normal(0, 1, N)

# ambient drone bed
drone = sum(np.sin(2 * np.pi * f * t + p) * a for f, a, p in [(55, .5, 0), (82.4, .3, 1), (110, .2, 2), (164.8, .08, 3)])
drone *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.17 * t)) * np.clip(t / 1.5, 0, 1) * np.clip((DUR - t) / 0.6, 0, 1)
add(lp(drone, 400), 0.10)

def boom(t0, f0=48, g=1.0, d=1.2):
    x = np.clip(t - t0, 0, None)
    ph = 2 * np.pi * (f0 * x + 30 * (1 - np.exp(-x * 12)) / 12)
    add(np.sin(ph) * env(t0, 0.005, d), 0.9 * g)
    add(lp(noise, 180) * env(t0, 0.002, 0.08), 0.6 * g)

def whoosh(t0, length, lo=300, hi=4000, g=0.5, rev=False, pan=0.0):
    x = np.clip((t - t0) / length, 0, 1)
    shape = np.sin(np.pi * x) ** 2 if not rev else x ** 3 * (t < t0 + length)
    shape *= (t >= t0) & (t <= t0 + length)
    sig = band(noise, lo, hi) * shape
    add(sig, g, pan)

def tone(t0, length, f0, f1, g=0.2, harm=(1, .5, .25), a=0.2, pan=0.0):
    x = np.clip(t - t0, 0, length)
    f = f0 + (f1 - f0) * (x / length)
    ph = 2 * np.pi * np.cumsum(f) / SR
    sig = sum(h * np.sin(ph * (i + 1)) for i, h in enumerate(harm))
    e = np.clip(x / a, 0, 1) * np.clip((length - x) / a, 0, 1) * ((t >= t0) & (t <= t0 + length))
    add(sig * e, g, pan)

def click(t0, g=0.25, f=3000, pan=0.0):
    add(band(noise, f * .6, min(f * 1.6, 20000)) * env(t0, 0.0005, 0.012), g, pan)

def ring(t0, g=0.3, d=0.9):
    for f, a in [(523, 1), (1187, .6), (2013, .4), (3155, .25)]:
        add(np.sin(2 * np.pi * f * np.clip(t - t0, 0, None)) * env(t0, 0.002, d * (1 - f / 6000)), g * a * 0.25)

def shimmer(t0, length, g=0.12):
    for f in (1760, 2217, 2637, 3520):
        add(np.sin(2 * np.pi * f * t) * env(t0, length * 0.3, length * 0.5) * (t < t0 + length * 2), g / 4,
            pan=np.sin(f) * 0.6)

# SCENE 1 — hook
boom(0.05, g=1.0, d=1.6)
whoosh(0.1, 1.4, 200, 3000, 0.55)
whoosh(0.25, 2.2, 2000, 9000, 0.12, pan=-0.5)          # beam sweep hiss L->R
whoosh(1.2, 1.5, 2000, 9000, 0.12, pan=0.5)
shimmer(0.9, 1.2, 0.08)

# SCENE 2 — levitation hum
tone(3.0, 3.2, 62, 104, 0.22, harm=(1, .6, .3, .15), a=0.5)
tone(3.0, 3.2, 124.5, 209, 0.05, a=0.5, pan=0.4)
whoosh(2.95, 0.8, 150, 1500, 0.35)

# SCENE 3 — explode: servo + slide + seat clicks per part
for i in range(10):
    t0 = 6.15 + 0.11 * i
    pan = (-1) ** i * 0.5
    tone(t0, 0.55, 380 + 30 * i, 820 + 30 * i, 0.035, harm=(1, .4, .3, .2, .1), a=0.08, pan=pan)
    whoosh(t0 + 0.1, 1.6, 600, 5000, 0.10, pan=pan)
    click(t0 + 2.1, 0.22, 2500 + 200 * i, pan)
tone(6.0, 4.0, 90, 96, 0.06, a=0.8)

# SCENE 4 — reassembly + lock
whoosh(10.0, 1.55, 300, 6000, 0.5, rev=True)
for i in range(10):
    tone(10.0 + 0.07 * i, 1.4, 900 - 25 * i, 420, 0.02, a=0.15, pan=(-1) ** i * 0.4)
boom(11.55, f0=58, g=0.9, d=0.5)
ring(11.55, 0.6)
for k in range(3): click(11.55 + 0.03 * k, 0.4, 1800)

# SCENE 5 — descent, impact, sweep, logo
tone(12.0, 1.2, 104, 58, 0.18, a=0.25)
boom(13.15, f0=42, g=1.1, d=1.8)
whoosh(13.15, 1.3, 2500, 12000, 0.14, pan=0.3)
shimmer(13.3, 1.0, 0.12)
whoosh(12.7, 0.7, 400, 4000, 0.25)
ring(12.8, 0.15, 1.5)

mix = np.stack([L, R], 1)
mix = np.tanh(mix / np.abs(mix).max() * 1.6) * 0.89
mix *= np.clip((DUR - t) / 0.4, 0, 1)[:, None]
with wave.open("audio.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("ok")
