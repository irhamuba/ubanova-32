"""Generate 32x32 pixel-art sprites with UbaNova-32 (CPU or GPU).

usage:
  python generate.py "iron sword" "red potion" --n 4
  python generate.py "green slime, on a light blue background" --bg
  python generate.py "wooden chest" --size 16                 # chunky 16x16 look (saved as 16x16)
  python generate.py "golden crown" --detail high --fill mid  # v1.1 controls
"""
import argparse
import time

from PIL import Image

from infer32 import SpriteGenerator32

LEVELS = ("auto", "any", "low", "mid", "high")


def main():
    ap = argparse.ArgumentParser(description="Text -> 32x32 pixel-art sprites with UbaNova-32")
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("--n", type=int, default=4, help="sprites per prompt")
    ap.add_argument("--size", type=int, choices=(16, 32), default=32, help="32 = native, 16 = chunky 16x16 look")
    ap.add_argument("--bg", action="store_true", help="fill the background (name a color in the prompt, e.g. 'on a blue background')")
    ap.add_argument("--detail", choices=LEVELS, default="auto", help="color count / shading (v1.1 models)")
    ap.add_argument("--fill", choices=LEVELS, default="auto", help="object size on the canvas (v1.1 models)")
    ap.add_argument("--cfg", type=float, default=2.0)  # v1.1 eval: least border-touching, sizes closest to real
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=None, help="default: weights/model32_v11.pt, else downloaded from Hugging Face")
    ap.add_argument("--palette", default=None)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--scale", type=int, default=8, help="upscale factor of the saved PNG")
    ap.add_argument("--out", default="sprites32.png")
    a = ap.parse_args()

    gen = SpriteGenerator32(a.model, a.palette, a.device)
    images = []
    t0 = time.time()
    for p in a.prompts:
        images += gen.generate(p, n=a.n, cfg=a.cfg, temperature=a.temperature, top_p=a.top_p, seed=a.seed,
                               size=a.size, background=a.bg, fill=a.fill, detail=a.detail)
    print(f"{len(images)} sprite(s) in {time.time() - t0:.1f}s on {gen.device} ({gen.model_path})")

    side, cols = images[0].size[0], a.n
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * side, rows * side), (0, 0, 0, 0))
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * side, (i // cols) * side))
    sheet.resize((cols * side * a.scale, rows * side * a.scale), Image.NEAREST).save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
