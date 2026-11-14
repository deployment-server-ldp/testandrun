# Mark 8 Automatic Cigarette Making Machine (MK-8) – reel

`out/mk8_exploded_view_reel.mp4` — 1080×1920, 44.1 s, 30 fps, H.264 + AAC music. Exploded view, spare-part labels, reassembly, then the four close-up sections from `source/mk8_reel_v3.mp4`, then the closing shot.

`out/cigarette_machine_exploded_view.mp4` — the earlier 25 s silent version.

Layered and re-renderable:

- `source/` – the original photos and infographic.
- `layers/machine_wide_cut.png` – machine photo with background removed (`build/cut.py`).
- `layers/parts/*.png` – one RGBA layer per component group, cut from the infographic (`build/extract.py`).
- `parts.json` – crop, exploded position, scale, assembled position, label text and callout target for every layer.
- `build/music.py` – re-cuts the music of `source/mk8_reel_v3.mp4` to the film length, hits placed on the explosion and scene changes.
- `render.py` – timing, camera, callouts, titles. `python3 render.py` renders the MP4; `python3 render.py --still 12.5` renders one frame.

Requires `opencv-python-headless numpy pillow scipy` (and `rembg` only to redo the photo cut-out), Inter fonts, ffmpeg.
