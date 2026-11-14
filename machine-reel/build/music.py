#!/usr/bin/env python3
"""Re-cut the music of source/mk8_reel_v3.mp4 (25 s, 120 BPM) to the length of the film.
usage: music.py <out.wav>
The track's big hits (1.0, 6.0, 20.5 s) are placed on the explosion, the first label,
the start of the close-up tour and the final shot; its held break covers the labels leaving."""
import os, subprocess, sys, wave, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SR = 48000
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{R}/source/mk8_reel_v3.mp4", "-vn", "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"],
                     capture_output=True, check=True).stdout
src = np.frombuffer(raw, "<i2").reshape(-1, 2).astype(np.float64) / 32768
# (film start, source start, length) - all on the 0.5 s beat grid, film grid offset 4.1 s
EDL = [(4.1, 1.0, 4.0),     # explosion  <- opening hit + beat
       (8.1, 6.0, 7.0),     # labels     <- second hit, beat, then the held break as the labels leave
       (15.1, 13.0, 2.5),   # reassembly <- beat returns
       (17.6, 18.0, 2.5),
       (20.1, 1.0, 19.5),   # close-up tour <- the music exactly as it was cut to this footage
       (39.6, 20.5, 4.5)]   # final shot <- last hit and the ringing tail
TOTAL = 44.1
PRE, XF = 0.015, 0.012      # splice just before the beat's transient, with a short crossfade
out = np.zeros((int(TOTAL * SR) + SR, 2))
def place(seg, at, fi, fo):
    seg = seg.copy(); n = len(seg)
    a, b = int(fi * SR), int(fo * SR)
    if a: seg[:a] *= np.linspace(0, 1, a)[:, None]
    if b: seg[-b:] *= np.linspace(1, 0, b)[:, None]
    i = int(round(at * SR)); out[i:i + n] += seg
for k, (ft, st, ln) in enumerate(EDL):
    last = k == len(EDL) - 1
    s0 = int(round((st - PRE) * SR)); s1 = int(round((st + ln - PRE + (0 if last else XF)) * SR))
    place(src[s0:min(s1, len(src))], ft - PRE, XF, 0 if last else XF)
# intro (0 - 4.1 s): the track's opening drone, extended, plus its final chord reversed as a swell into the hit
dr = src[int(0.30 * SR):int(0.96 * SR)]; n = len(dr); xf = int(0.25 * SR)
w = np.ones(n); w[:xf] = np.sin(np.linspace(0, np.pi / 2, xf)) ** 2; w[-xf:] = np.cos(np.linspace(0, np.pi / 2, xf)) ** 2
pos = 0
while pos < int(4.1 * SR):
    seg = dr * w[:, None]; e = min(pos + n, int(4.1 * SR))
    out[pos:e] += seg[:e - pos] * 1.0
    pos += n - xf
env = np.clip(np.arange(int(4.1 * SR)) / (1.2 * SR), 0, 1)
out[:len(env)] *= env[:, None]
tail = src[int(23.05 * SR):int(24.95 * SR)][::-1].copy()
tail *= (np.linspace(0, 1, len(tail)) ** 1.5)[:, None]
tail[-int(.02 * SR):] *= np.linspace(1, 0, int(.02 * SR))[:, None]
i = int((4.1 - PRE) * SR) - len(tail); out[i:i + len(tail)] += tail * 0.8
out = out[:int(TOTAL * SR)]
out[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))[:, None]
out = np.clip(out * 0.9, -1, 1)
with wave.open(sys.argv[1], "wb") as wv:
    wv.setnchannels(2); wv.setsampwidth(2); wv.setframerate(SR); wv.writeframes((out * 32767).astype("<i2").tobytes())
