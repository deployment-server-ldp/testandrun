from rembg import remove, new_session
from PIL import Image
import sys
s = new_session("isnet-general-use")
for n in ["machine_wide","machine_front"]:
    im = Image.open(f"source/{n}.jpg").convert("RGB")
    out = remove(im, session=s)
    out.save(f"layers/{n}_cut.png")
    bg = Image.new("RGB", out.size, (8,14,30)); bg.paste(out, mask=out.split()[3]); bg.save(f"build/{n}_prev.jpg")
