# Cigarette Making Machine – exploded view reel

`out/cigarette_machine_exploded_view.mp4` — 1080×1920, 25 s, 30 fps, H.264.

Layered and re-renderable:

- `source/` – the original photos and infographic.
- `layers/machine_wide_cut.png` – machine photo with background removed (`build/cut.py`).
- `layers/parts/*.png` – one RGBA layer per component group, cut from the infographic (`build/extract.py`).
- `parts.json` – crop, exploded position, scale, assembled position, label text and callout target for every layer.
- `render.py` – timing, camera, callouts, titles. `python3 render.py` renders the MP4; `python3 render.py --still 12.5` renders one frame.

Requires `opencv-python-headless numpy pillow scipy` (and `rembg` only to redo the photo cut-out), Inter fonts, ffmpeg.
