---
name: reel-transcription
description: Word-timed English transcripts for talking-head reels (Durrani's Dental Clinic, Dr. Shoaib Durrani, Dr Marium, Dr Basma, Glance) using NVIDIA Parakeet TDT 0.6B v2 through sherpa-onnx on CPU. Produces words.json for the caption engines plus a timed .txt and an .srt, with name fixes, phrase-start snapping and re-checks of single stretches. Use whenever a reel, VO or interview needs a transcript, word timings, captions or subtitles, or a stretch must be re-checked after a cut.
---

# Reel transcription (word timings for captions)

This is the transcription used for the Durrani's Dental Clinic style reels. Plain mode gives exactly the same `words.json` as the pipeline that made the Dr Marium reels the user approved in October 2026 (checked word for word on test files). The `--snap`, `--fix` and `--window` options, the `.txt` and the `.srt` were added on top and tested. They never change plain mode.

Its output is the input of `aligner-captions` and of the Durrani "Script 3" caption engine. Both read `[{"w": word, "t0": start_s, "t1": end_s}, ...]`. (`aligner-captions` mentions pocketsphinx alignment for when you only have a script. When you have the audio, use this skill.)

## 1. Set up the engine (once per workspace, about 2 minutes, about 700 MB)

```bash
A=${ASR_DIR:-/home/claude/asr}
mkdir -p "$A" && cd "$A" || exit 1
if [ ! -f READY ]; then
  curl -fsSL -o bin.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-linux-x64-static.tar.bz2 &&
  curl -fsSL -o pk.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2 &&
  tar xjf bin.tar.bz2 sherpa-onnx-v1.13.8-linux-x64-static/bin/sherpa-onnx-offline &&
  tar xjf pk.tar.bz2 &&
  rm -f bin.tar.bz2 pk.tar.bz2 && touch READY
fi
[ -f READY ] && echo "ASR ready in $A" || echo "ASR setup FAILED in $A"
```

- The block skips itself once `READY` exists. Only the one binary that is needed is unpacked (35 MB instead of 1.3 GB). It needs no other files.
- The engine comes from GitHub release downloads, which the workspace can reach. huggingface.co and the OpenAI Whisper model host were blocked (403), so the usual Whisper / faster-whisper model downloads fail.
- Run it in the background while you look at the footage. Don't transcribe until it prints `ASR ready`. In the Marium session the first run failed because the model was still being unpacked. The script also refuses to start while files are missing.
- If a download fails, run `curl -sS "$HTTPS_PROXY/__agentproxy/status"`, read `/root/.ccr/README.md`, and tell the user. Never turn off TLS checks.
- A folder from an earlier session (for example `/home/claude/asr` without `READY`) works as is. The script only needs the binary and the four model files.

## 2. Save the script

Write the code block at the end of this file, unchanged, to `/home/claude/asr/transcribe_words.py`. Smoke test (expected text: "Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait."):

```bash
python3 /home/claude/asr/transcribe_words.py /home/claude/asr/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav -o /home/claude/asr/smoke/0
```

## 3. Transcribe

Put the input file first, then the options:

```bash
mkdir -p /home/claude/work/audio
python3 /home/claude/asr/transcribe_words.py /mnt/user-data/uploads/Marium_4.mp4 -o /home/claude/work/audio/Marium_4 --snap
```

- Input: any video or audio file that ffmpeg reads (camera MP4/MOV, WAV, the VO). Use the clean voice track, not an export with music under it.
- Output: `Marium_4.words.json` for the caption engines, `Marium_4.txt` for proofreading (one timed line per cue), and `Marium_4.srt` for Premiere / AE caption tracks. It also prints the chunk cuts and a `word@time` list, which is what you need to write the caption spec.
- Speed: about half the clip length on 2 CPU cores (27.7 s took 13 s, 83 s took 40 s). Every chunk reloads the model (about 2.5 s).
- How it works (keep it this way; it is what the approved reels used):
  - Audio goes to 16 kHz mono.
  - `silencedetect=n=-35dB:d=0.18` finds the pauses. A cut goes in the middle of a pause, at least 6 s after the previous cut and more than 2 s before the end.
  - Each chunk is decoded on its own (`--model-type=nemo_transducer`, 2 threads). Chunk times are kept at full precision. Rounding them to milliseconds moves the model's 80 ms frame grid, and that changed name spellings and timings by up to 0.16 s in testing.
  - A token that starts with a space starts a new word. Sub-words and punctuation join the word before them, so "Karachi." stays one word.
  - Every word lasts at least 0.08 s and never runs into the next one.
  - stdout and stderr are read separately. If both go to one file (`2>&1`), a long result gets cut in two by log lines and the JSON breaks. That happened in the Marium session and again in testing.

## 4. Timing accuracy and `--snap`

Measured on test speech with exact word onsets:

- Inside a phrase, word starts were within 0.15 s (mean 0.06 s). The model works in 80 ms steps.
- **The first word after a pause comes out early, by 0.06 to 0.45 s (mean about 0.2 s).** Captions then appear before the voice at the start of a line.
- `--snap` moves each phrase start to the real voice onset, measured from the waveform. A word is only moved when there is a real pause before it. Words never move earlier. Following words are pushed only as much as needed to keep them 0.08 s apart.
- Results with `--snap`:
  - Clean audio: phrase starts within 0.01 s.
  - With room noise and hum: within 0.05 s.
  - With a music bed: nothing changes, because the voice isn't 25 dB clear of the floor.
- Use `--snap` for captions, cards and cuts that land on words. Without it you get exactly the timings of the first Dr Marium reels (Marium 1–3).
- The last word before a pause is the least stable. Its time can move by about 0.3 s between runs. Caption group ends add 0.7–1.0 s after it anyway.

## 5. Proofread before anything uses the words (never skip)

The model does not know the script. The same audio can come out spelled differently on different runs or after a cut. Heard on these reels and in testing:

| Said | Parakeet wrote |
|---|---|
| Marium | Mariam, Miriam, Merriam (not once correct so far) |
| Durrani's Dental Clinic | Irani's dental clinic (sometimes right) |
| Shoaib | Shawib, Showib, Shoib |
| Dr. | Doctor, Dr, Dr. (changes between runs) |
| 5, 15 | five, fifteen |
| it's | is |
| I'm, you've | I am, you have (listen for which one was said) |
| aren't | are (**the meaning flips; check every negative**) |
| gums | gum |
| health | heal |

1. Read `NAME.txt` from start to end. Compare it with the user's script or storyboard if there is one. If the audio clearly differs from the script, follow the audio and tell the user.
2. **Names and brand words:** run again with `--fix`. It replaces whole words, ignores case and keeps punctuation, in all three outputs. Timings don't change. It prints how many times each fix was used, and `NO MATCH` for a typo. Use it only for names, because it changes every occurrence (`dental=Dental` would also change "your last dental visit"). Example: `--fix Mariam=Marium Miriam=Marium "Irani's=Durrani's" Shawib=Shoaib Showib=Shoaib`.
3. **Everything else** (numbers, contractions, a lost negative, plurals): leave `words.json` as the model heard it. Fix what is displayed in the caption spec, with the engine's display=spoken syntax, as was done on the Marium reels: `5=five`, `aren't=are`, `health=heal`. To show one word over two spoken ones, the Script 3 engine uses `I'm=I -=am`.
4. **An unclear stretch:** re-read it with `--window START END`, trying two different windows a few seconds wide, and compare. Check the pauses in that stretch, in source time, with `ffmpeg -i X -af "atrim=A:B,silencedetect=n=-35dB:d=0.15" -f null - 2>&1 | grep silence_`. Look at a contact sheet of the frames to see whether the mouth moves. Never invent words. If a line is missing or garbled, tell the user. Marium 3 had a 5.5 s silent gap at 14.5–20.5 s; the check-up line was missing from the edit, and the user was asked for the take.
5. List every text correction in one line of your reply so the user can check it. Example: "captions use Dr. Marium (your spelling), oral health (audio sounds like oral heal), aren't always visible".

## 6. After cutting the video

The render drops whole frames. Move the words the same way and keep the cut list next to the render code:

```python
def cut_words(W, cuts, fps=30):
    """Take the stretches cuts=[(a, b), ...] (source seconds) out of a word list the way the render drops
    frames: both ends snap to the frame grid, words starting inside a cut go, later words move back."""
    cuts = sorted((round(a * fps) / fps, round(b * fps) / fps) for a, b in cuts)
    out = []
    for w in W:
        if any(a <= w['t0'] < b for a, b in cuts): continue
        s = sum(b - a for a, b in cuts if w['t0'] >= b)
        out.append(dict(w, t0=round(w['t0'] - s, 3), t1=round(w['t1'] - s, 3)))
    for x, y in zip(out, out[1:]): x['t1'] = min(x['t1'], y['t0'])
    return out
```

In testing, two cuts gave the same word count as a fresh transcript of the cut audio, and the phrase start after a cut matched within 0.01 s. After rendering, check every seam by reading about 1.8 s around it on the export, for example `--window 13.6 15.4` on the delivered MP4. Every word on both sides should be complete.

## 7. Limits

- English only (Parakeet TDT 0.6B v2). Urdu or mixed Urdu-English speech will come out wrong. Tell the user instead of captioning it.
- Music under the voice (a finished export, for example): pauses aren't found, so it becomes one long chunk, and `--snap` does nothing. Transcribe the camera audio or the VO.
- One voice track, no speaker labels. For Glance street interviews, check who says what from the picture.
- No hotwords or custom vocabulary. Names are fixed after reading, in step 5.

## transcribe_words.py

```python
#!/usr/bin/env python3
"""transcribe_words.py - word-timed English transcript for talking-head reels.

Engine: sherpa-onnx v1.13.8 offline CLI + NVIDIA Parakeet TDT 0.6B v2 (int8), CPU only.
The audio is split at silences (cuts land mid-pause, more than 6 s apart),
each chunk is decoded on its own, and the token times are shifted back to the full clip.

  python3 transcribe_words.py VIDEO_OR_AUDIO [-o OUT_BASE]
      writes OUT_BASE.words.json  [{"w": word, "t0": start_s, "t1": end_s}, ...]  (caption engines)
             OUT_BASE.txt         transcript, one timed line per subtitle cue          (proofreading)
             OUT_BASE.srt         subtitles                                            (Premiere / AE)
      OUT_BASE defaults to ./<input file name without extension>

  options:  --snap               first word after each pause moved to the measured voice onset
            --fix WRONG=RIGHT ...  whole-word text fixes in every output (names: Mariam=Marium)
            --threads N          CPU threads for the engine (default 2)

  python3 transcribe_words.py VIDEO_OR_AUDIO --window 13.4 17.2
      re-reads one stretch only (times are seconds in the input); prints, writes nothing
"""
import argparse, json, os, re, subprocess, sys, tempfile

ASR = os.environ.get('ASR_DIR', '/home/claude/asr')
BIN = f'{ASR}/sherpa-onnx-v1.13.8-linux-x64-static/bin/sherpa-onnx-offline'
MOD = f'{ASR}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8'
NEEDED = [BIN] + [f'{MOD}/{n}' for n in ('encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt')]


def to_wav(src, dst, a=None, b=None):
    """16 kHz mono 16-bit WAV, optionally only the stretch a..b (seconds in src)."""
    cmd = ['ffmpeg', '-v', 'error', '-y']
    if a is not None: cmd += ['-ss', str(a)]      # full precision: rounding the cut shifts the model's
    if b is not None: cmd += ['-to', str(b)]      # 80 ms frame grid and can change uncertain words
    cmd += ['-i', src, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(dst):
        sys.exit(f'ffmpeg could not read audio from {src}:\n{r.stderr[-800:]}')


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def chunk_points(wav, dur):
    """Cut in the middle of pauses (silencedetect -35 dB, 0.18 s); a cut needs >6 s since the
    previous cut and >2 s before the end."""
    log = subprocess.run(['ffmpeg', '-i', wav, '-af', 'silencedetect=n=-35dB:d=0.18', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    pauses, start = [], None
    for line in log.splitlines():
        m = re.search(r'silence_start: (-?[\d.]+)', line)
        if m: start = max(0.0, float(m.group(1)))
        m = re.search(r'silence_end: (-?[\d.]+)', line)
        if m and start is not None:
            pauses.append((start, float(m.group(1)))); start = None
    cuts = [0.0]
    for a, b in pauses:
        mid = (a + b) / 2
        if mid - cuts[-1] > 6 and dur - mid > 2: cuts.append(mid)
    cuts.append(dur)
    return cuts


def decode(wav, threads):
    """Run Parakeet on one WAV; return its result JSON. stdout and stderr are captured
    separately: merged into one stream (2>&1), a long result gets cut in two by log lines."""
    r = subprocess.run([BIN, f'--encoder={MOD}/encoder.int8.onnx', f'--decoder={MOD}/decoder.int8.onnx',
                        f'--joiner={MOD}/joiner.int8.onnx', f'--tokens={MOD}/tokens.txt',
                        '--model-type=nemo_transducer', f'--num-threads={threads}', wav],
                       capture_output=True, text=True)
    lines = [l for l in (r.stdout + '\n' + r.stderr).splitlines() if l.startswith('{"lang"')]
    if r.returncode != 0 or not lines:
        sys.exit(f'sherpa-onnx failed on {wav} (exit {r.returncode}):\n' + r.stderr[-1500:])
    return json.loads(lines[0])


def tokens_to_words(j, offset):
    """A token that starts with a space starts a new word; the rest (sub-words, punctuation) join it."""
    toks, ts = j.get('tokens', []), j.get('timestamps', [])
    durs = j.get('durations') or [0.08] * len(toks)
    words, cur = [], None
    for tok, t, d in zip(toks, ts, durs):
        if tok.startswith(' ') or cur is None:
            if cur: words.append(cur)
            cur = {'w': tok.strip(), 't0': round(offset + t, 3), 't1': round(offset + t + d, 3)}
        else:
            cur['w'] += tok; cur['t1'] = round(offset + t + d, 3)
    if cur: words.append(cur)
    return [w for w in words if w['w']]


def tidy(words):
    """Every word lasts at least 0.08 s but never runs into the next word."""
    for x, y in zip(words, words[1:]):
        x['t1'] = min(max(x['t1'], x['t0'] + 0.08), y['t0'])
    return words


def snap_onsets(words, wav, look_ahead=0.45, rise_db=12.0):
    """Parakeet stamps the first word after a pause early (measured 0.06-0.45 s). For each word, look from
    t0-0.1 to t0+0.45 s for the first 10 ms frame that is 12 dB over the local noise floor for 30 ms and
    sits after a real pause: the 120 ms before it (not counting the last 30 ms, where a soft attack can
    start) stay near the floor. If that onset is later than t0, the word moves there; following words that
    would now overlap are pushed just enough to keep 0.08 s each. Words never move earlier. Skipped where
    the voice is not 25 dB clear of the floor (music bed, loud room)."""
    import numpy as np
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', wav, '-f', 'f32le', '-ac', '1', '-ar', '16000', '-'],
                         capture_output=True, check=True).stdout
    y = np.frombuffer(raw, np.float32)
    n = len(y) // 160
    db = 20 * np.log10(np.sqrt((y[:n * 160].reshape(n, 160) ** 2).mean(1)) + 1e-9)    # 10 ms frames
    moved = 0
    for i, w in enumerate(words):
        c0, c1 = int(max(0.0, w['t0'] - 1.0) * 100), min(n, int((w['t0'] + 1.0) * 100))
        lo, hi = int(max(0.0, w['t0'] - 0.1) * 100), min(n - 3, int((w['t0'] + look_ahead) * 100))
        if c1 - c0 < 50 or hi <= lo: continue
        floor, peak = np.percentile(db[c0:c1], 10), np.percentile(db[c0:c1], 95)
        if peak - floor < 25: continue
        k = next((k for k in range(max(lo, 15), hi) if (db[k:k + 3] > floor + rise_db).all()), None)
        if k is None or k / 100 <= w['t0'] + 0.02: continue
        pre = db[k - 15:k - 3]                     # 120 ms before the onset, leaving 30 ms for a soft attack
        if np.median(pre) > floor + rise_db / 2 or pre.max() > floor + rise_db: continue
        w['t0'] = round(k / 100, 3); moved += 1
        for a, b in zip(words[i:], words[i + 1:]):
            if b['t0'] >= a['t0'] + 0.08: break
            b['t0'] = round(a['t0'] + 0.08, 3)
    for w in words: w['t1'] = max(w['t1'], round(w['t0'] + 0.08, 3))
    tidy(words)
    return moved


def apply_fixes(words, fixes):
    """WRONG=RIGHT whole-word replacements (case-insensitive, punctuation kept): Mariam=Marium."""
    table = {}
    for f in fixes:
        wrong, _, right = f.partition('=')
        if not wrong.strip() or not right.strip() or ' ' in right.strip():
            sys.exit(f'--fix takes WRONG=RIGHT with one word on each side, got {f!r}')
        table[wrong.strip().lower()] = right.strip()
    hits = dict.fromkeys(table, 0)
    for w in words:
        lead, core, tail = re.fullmatch(r"([^\w']*)(.*?)([^\w']*)", w['w']).groups()
        if core.lower() in table:
            w['w'] = lead + table[core.lower()] + tail; hits[core.lower()] += 1
    for k, n in hits.items():
        print(f'fix {k} -> {table[k]}: {n}x' + ('' if n else '   NO MATCH, check the spelling'))


ABBREV = ('dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'st.', 'vs.')


def cues(words, max_chars=42, max_gap=0.6):
    """Group words into subtitle cues: break after . ? ! (not Dr. etc.), at pauses over 0.6 s, or past 42 characters."""
    groups, cur = [], []
    for w in words:
        if cur and (len(' '.join(x['w'] for x in cur + [w])) > max_chars
                    or w['t0'] - cur[-1]['t1'] > max_gap
                    or (cur[-1]['w'].endswith(('.', '?', '!')) and cur[-1]['w'].lower() not in ABBREV)):
            groups.append(cur); cur = []
        cur.append(w)
    if cur: groups.append(cur)
    out = []
    for i, g in enumerate(groups):
        nxt = groups[i + 1][0]['t0'] if i + 1 < len(groups) else float('inf')
        t0 = g[0]['t0']
        t1 = min(max(g[-1]['t1'] + 0.2, t0 + 0.6), nxt)
        out.append((t0, t1, ' '.join(x['w'] for x in g)))
    return out


def srt_time(t):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def mmss(t):
    cs = int(round(t * 100)); m, cs = divmod(cs, 6000)
    return f'{m:02d}:{cs // 100:02d}.{cs % 100:02d}'


def main():
    ap = argparse.ArgumentParser(description='Word-timed transcript (Parakeet TDT 0.6B v2 via sherpa-onnx).')
    ap.add_argument('input', help='video or audio file')
    ap.add_argument('-o', '--out', help='output base path (default: ./<input name>)')
    ap.add_argument('--window', nargs=2, type=float, metavar=('START', 'END'),
                    help='re-read only this stretch (seconds in the input); prints, writes no files')
    ap.add_argument('--snap', action='store_true',
                    help='move the first word after each pause to the measured voice onset')
    ap.add_argument('--fix', nargs='+', default=[], metavar='WRONG=RIGHT',
                    help='whole-word text fixes applied to all outputs, e.g. Mariam=Marium "Irani\'s=Durrani\'s"')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    if args.window and args.snap:
        sys.exit('--snap works on full transcripts only, not with --window')

    missing = [p for p in NEEDED if not os.path.isfile(p)]
    if missing:
        sys.exit('Speech engine not set up (missing: ' + ', '.join(missing) + '). Run the setup step of the skill first.')
    if not os.path.isfile(args.input):
        sys.exit(f'No such file: {args.input}')

    with tempfile.TemporaryDirectory() as tmp:
        if args.window:
            a, b = args.window
            to_wav(args.input, f'{tmp}/win.wav', a, b)
            words = tidy(tokens_to_words(decode(f'{tmp}/win.wav', args.threads), a))
            if args.fix: apply_fixes(words, args.fix)
            print(' '.join(w['w'] for w in words))
            print(' '.join(f"{w['w']}@{w['t0']:.2f}" for w in words))
            return

        to_wav(args.input, f'{tmp}/full.wav')
        dur = duration(f'{tmp}/full.wav')
        cuts = chunk_points(f'{tmp}/full.wav', dur)
        words = []
        for a, b in zip(cuts, cuts[1:]):
            to_wav(f'{tmp}/full.wav', f'{tmp}/chunk.wav', a, b)
            words += tokens_to_words(decode(f'{tmp}/chunk.wav', args.threads), a)
        words = tidy(words)
        moved = snap_onsets(words, f'{tmp}/full.wav') if args.snap else None
    if args.fix: apply_fixes(words, args.fix)

    base = args.out or os.path.splitext(os.path.basename(args.input))[0]
    if os.path.dirname(base): os.makedirs(os.path.dirname(base), exist_ok=True)
    with open(base + '.words.json', 'w') as f:
        json.dump(words, f, indent=0)
    cs = cues(words)
    with open(base + '.txt', 'w') as f:
        for t0, t1, text in cs:
            f.write(f'[{mmss(t0)}] {text}\n')
    with open(base + '.srt', 'w') as f:
        for i, (t0, t1, text) in enumerate(cs, 1):
            f.write(f'{i}\n{srt_time(t0)} --> {srt_time(t1)}\n{text}\n\n')

    print(f'{len(words)} words, {dur:.2f} s, chunk cuts at ' + ', '.join(f'{c:.2f}' for c in cuts)
          + ('' if moved is None else f', {moved} phrase starts snapped to the voice onset'))
    print(f'wrote {base}.words.json, {base}.txt, {base}.srt')
    print(' '.join(f"{w['w']}@{w['t0']:.2f}" for w in words))


if __name__ == '__main__':
    main()
```
