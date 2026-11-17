"""Build the UbaNova-32 training data (CPU kernel).

Sources
  azyg0s/pixelart-172k               32x32 RGBA icons -> main 32x32 set (+ its val splits for eval)
  ProbablyHumanAB/moss-pixelart-64   64x64 RGBA sprites -> downscaled to 32x32
  Angais/krea-2-pixelart-16x16-synthetic  16x16 sprites -> 16x16 set (atlas 1.4m is read by the trainer)
Every sprite is mapped to the UbaNova palette (palette.npy from the atlas, index 0 = transparent),
with at most MAX_COLORS colors per sprite. Backgrounds are not baked in: the trainer adds them.

Output (/kaggle/working):
  data32.npz      tokens (N, 1024) uint16, split (N,) uint8 [0 train, 1+ eval splits], source (N,) uint8
  captions32.json captions for data32
  data16_extra.npz / captions16_extra.json   krea sprites (N, 256)
  palette.npy, prep_report.json, compare_*.png
"""
import glob
import io
import json
import os
import re
import zipfile

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from scipy.spatial import cKDTree

OUT = "/kaggle/working"
MAX_COLORS = 32
rng = np.random.default_rng(0)
R = {}

z = zipfile.ZipFile(hf_hub_download("Angais/pixel-art-atlas-1.4m", "pixel-art-16x16-1.4m.zip", repo_type="dataset"))
palette = np.load(z.open("palette.npy")).astype(np.uint8)
np.save(f"{OUT}/palette.npy", palette)
tree = cKDTree(palette[1:].astype(np.float32))  # index 0 is transparent, never a color target


def to_tokens(rgba):
    """(H, W, 4) uint8 -> (H*W,) palette indices, alpha < 128 = transparent, <= MAX_COLORS colors."""
    a = rgba[..., 3] >= 128
    rgb = rgba[..., :3][a]
    out = np.zeros(a.shape, np.uint16)
    if len(rgb) == 0:
        return out.ravel(), 0
    uniq = np.unique(rgb, axis=0)
    if len(uniq) > MAX_COLORS:
        q = Image.fromarray(rgb[None]).quantize(colors=MAX_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        rgb = np.asarray(q.convert("RGB"))[0]
    out[a] = tree.query(rgb.astype(np.float32))[1] + 1
    return out.ravel(), len(uniq)


def downscale2(rgba):
    """64 -> 32 by taking the most common color of each 2x2 block (keeps hard pixel edges)."""
    h, w = rgba.shape[:2]
    b = rgba.reshape(h // 2, 2, w // 2, 2, 4).transpose(0, 2, 1, 3, 4).reshape(h // 2, w // 2, 4, 4)
    out = np.zeros((h // 2, w // 2, 4), np.uint8)
    for i in range(h // 2):
        for j in range(w // 2):
            px = [tuple(p) if p[3] >= 128 else (0, 0, 0, 0) for p in b[i, j]]
            out[i, j] = max(set(px), key=px.count)
    return out


def render(tok, side):
    t = tok.reshape(side, side)
    return np.concatenate([palette[t], ((t != 0) * 255).astype(np.uint8)[..., None]], -1)


def sheet(rows, path, cell=64):
    """rows: list of lists of RGBA arrays -> one PNG on gray."""
    cols = max(len(r) for r in rows)
    img = Image.new("RGBA", (cols * cell, len(rows) * cell), (128, 128, 128, 255))
    for y, r in enumerate(rows):
        for x, a in enumerate(r):
            img.alpha_composite(Image.fromarray(a).resize((cell, cell), Image.NEAREST), (x * cell, y * cell))
    img.convert("RGB").save(path)


# ---------------------------------------------------------------- azyg0s 32x32
root = snapshot_download("azyg0s/pixelart-172k", repo_type="dataset", allow_patterns=["data/*.parquet"])
SPLITS = ["train", "val", "val_patterns", "val_r6", "val_binding", "val_unseen"]
toks, caps, split, src, ncol = [], [], [], [], []
flags_seen, skipped = {}, 0
before, after = [], []
for f in sorted(glob.glob(f"{root}/data/*.parquet")):
    s = re.match(r"(.+?)-\d{5}-of-", os.path.basename(f)).group(1)
    for row in pq.read_table(f).to_pylist():
        fl = row.get("flags") or []
        for x in fl:
            flags_seen[x] = flags_seen.get(x, 0) + 1
        if s == "train" and fl:
            skipped += 1
            continue
        rgba = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGBA"))
        if rgba.shape[:2] != (32, 32):
            skipped += 1
            continue
        t, n = to_tokens(rgba)
        if (t != 0).sum() < 16:
            skipped += 1
            continue
        toks.append(t); caps.append(row["text"] or row["caption"]); split.append(SPLITS.index(s)); src.append(0); ncol.append(n)
        if len(before) < 24 and s == "train" and n > MAX_COLORS:
            before.append(rgba); after.append(render(t, 32))
R["azyg0s"] = {"kept": len(toks), "skipped": skipped, "flags": flags_seen,
               "split_counts": {SPLITS[i]: int(np.sum(np.array(split) == i)) for i in range(len(SPLITS))},
               "orig_colors_mean": float(np.mean(ncol)), "frac_over_cap": float(np.mean(np.array(ncol) > MAX_COLORS))}
sheet([before[:12], after[:12], before[12:24], after[12:24]], f"{OUT}/compare_azyg0s.png")
print(R["azyg0s"], flush=True)

# ---------------------------------------------------------------- moss 64 -> 32
root = snapshot_download("ProbablyHumanAB/moss-pixelart-64", repo_type="dataset")
mb, ma, n_m = [], [], 0
for m in glob.glob(f"{root}/*/metadata.jsonl"):
    for line in open(m, encoding="utf-8"):
        r = json.loads(line)
        rgba = np.asarray(Image.open(os.path.join(os.path.dirname(m), r["file_name"])).convert("RGBA"))
        if rgba.shape[:2] != (64, 64):
            continue
        small = downscale2(rgba)
        t, _ = to_tokens(small)
        text = r["text"]
        mm = re.search(r"depicting (.+?)(\.|$)", text)
        cap = (mm.group(1) if mm else text).strip()
        cap = re.sub(r"^(a|an|the) ", "", cap)
        toks.append(t); caps.append(cap); split.append(0); src.append(1); n_m += 1
        if len(mb) < 12:
            mb.append(rgba); ma.append(render(t, 32))
R["moss"] = {"kept": n_m, "caption_examples": caps[-5:]}
sheet([mb, ma], f"{OUT}/compare_moss.png")

np.savez_compressed(f"{OUT}/data32.npz", tokens=np.stack(toks).astype(np.uint16), split=np.array(split, np.uint8), source=np.array(src, np.uint8))
json.dump(caps, open(f"{OUT}/captions32.json", "w"))

# ---------------------------------------------------------------- krea 16x16
root = snapshot_download("Angais/krea-2-pixelart-16x16-synthetic", repo_type="dataset")
k_t, k_c = [], []
for line in open(glob.glob(f"{root}/**/metadata.jsonl", recursive=True)[0], encoding="utf-8"):
    r = json.loads(line)
    p = os.path.join(root, r["sprite"])
    if not os.path.exists(p):
        continue
    t, _ = to_tokens(np.asarray(Image.open(p).convert("RGBA")))
    k_t.append(t); k_c.append(r["description"])
np.savez_compressed(f"{OUT}/data16_extra.npz", tokens=np.stack(k_t).astype(np.uint16))
json.dump(k_c, open(f"{OUT}/captions16_extra.json", "w"))
R["krea"] = {"kept": len(k_t), "caption_examples": k_c[:5]}

idx = rng.choice(np.flatnonzero(np.array(split) == 0), 48, replace=False)
sheet([[render(toks[i], 32) for i in idx[r * 12:(r + 1) * 12]] for r in range(4)], f"{OUT}/samples_train32.png")
R["total_32"] = len(toks)
R["caption_examples_32"] = [caps[i] for i in idx[:12]]
json.dump(R, open(f"{OUT}/prep_report.json", "w"), indent=1)
print(json.dumps(R, indent=1), flush=True)
