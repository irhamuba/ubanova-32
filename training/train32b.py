"""UbaNova-32 v1.1: fine-tune v1 with FILL (object size) and DETAIL (color count) controls.

Starts from model32.pt (ubanova-train32 output), 1 epoch at a low learning rate. Each sprite gets
fill/detail bins (terciles of the 32x32 training set); 20% of the time the bins are hidden (0 = any)
so the model still works without them. Original v1 notes follow.

Fine-tune UbaNova-16 into UbaNova-32 on a Kaggle T4.

Inputs (attached): ubanova-16-v1-backup (model.pt, train.py), ubanova32-code (ubanova32.py),
kernel output ubanova-prep32 (data32.npz, captions32.json, data16_extra.npz, captions16_extra.json).
Each batch mixes native 32x32 sprites (SIZE32) with 16x16 atlas/krea sprites upscaled 2x (SIZE16).
BG_PROB of samples get a solid background (BG_ON); 75% of those name the color in the caption.
Writes ckpt32b.pt (resumable), model32_v11.pt (fp16), samples_*.png and train_log.json to /kaggle/working.
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]


sys.path.insert(0, os.path.dirname(find("ubanova32.py")))
sys.path.insert(0, os.path.dirname(find("train.py")))
import train as T16  # noqa: E402  (data download, CLIP helpers)
import ubanova32 as U  # noqa: E402

OUT = "/kaggle/working"
dev = "cuda"
BATCH = 32
FRAC16 = 0.3
BG_PROB = 0.3
NAME_BG = 0.75
EPOCHS = 1
LR, WARMUP = 5e-5, 200
COND_DROP = 0.2
CFG_DROP = 0.1
CKPT_S = 1800
BUDGET_S = 11 * 3600
t_start = time.time()
torch.manual_seed(0)

# ---------------------------------------------------------------- data
T16.prepare()  # atlas 16x16 tokens + CLIP ids, in T16.DATA
tok16 = np.load(f"{T16.DATA}/tokens.npy")
ids16 = np.load(f"{T16.DATA}/text_ids.npy")[:-1]
split16 = np.load(f"{T16.DATA}/split.npy")
palette = np.load(f"{T16.DATA}/palette.npy")
tokenizer, clip = T16.load_clip(dev)

d32 = np.load(find("data32.npz"))
caps32 = json.load(open(find("captions32.json")))
ex = np.load(find("data16_extra.npz"))
caps_ex = json.load(open(find("captions16_extra.json")))
tr32 = np.flatnonzero(d32["split"] == 0)
ev32 = {name: np.flatnonzero(d32["split"] == i) for i, name in enumerate(["train", "val", "val_patterns", "val_r6", "val_binding", "val_unseen"]) if i}
ids32 = T16.encode_text(tokenizer, caps32).numpy()
ids_ex = T16.encode_text(tokenizer, caps_ex).numpy()
EOS = int(tokenizer.eos_token_id)

img32 = torch.from_numpy(d32["tokens"].astype(np.int16)).to(dev)
img16 = torch.from_numpy(np.concatenate([tok16[split16 == 0], ex["tokens"]]).astype(np.int16)).to(dev)
txt16 = np.concatenate([ids16[split16 == 0], ids_ex])
print(f"32x32 train {len(tr32):,}  16x16 train {len(img16):,}  eval {[(k, len(v)) for k, v in ev32.items()]}", flush=True)

@torch.no_grad()
def sprite_stats(t):
    """(N, P) tokens on GPU -> opaque fraction and number of distinct colors per sprite."""
    fill, ncol = [], []
    for i in range(0, len(t), 8192):
        x = t[i : i + 8192].long()
        fill.append((x != 0).float().mean(1))
        s = x.sort(1).values
        ncol.append(((s[:, 1:] != s[:, :-1]) & (s[:, 1:] != 0)).sum(1) + (s[:, 0] != 0).long())
    return torch.cat(fill), torch.cat(ncol).float()


fill32, ncol32 = sprite_stats(img32)
fill16, ncol16 = sprite_stats(img16)
tr32_t = torch.as_tensor(tr32, device=dev)
FILL_Q = torch.quantile(fill32[tr32_t], torch.tensor([1 / 3, 2 / 3], device=dev))
DETAIL_Q = torch.quantile(ncol32[tr32_t], torch.tensor([1 / 3, 2 / 3], device=dev))
binize = lambda v, q: torch.bucketize(v, q) + 1  # noqa: E731  (1 low, 2 mid, 3 high)
fill_b32, detail_b32 = binize(fill32, FILL_Q), binize(ncol32, DETAIL_Q)
fill_b16, detail_b16 = binize(fill16, FILL_Q), binize(ncol16, DETAIL_Q)
print(f"fill thresholds {FILL_Q.tolist()}  detail thresholds {DETAIL_Q.tolist()}", flush=True)

pal_tree = torch.from_numpy(palette[1:].astype(np.float32)).to(dev)
bg_names = list(U.BG_COLORS)
bg_idx = torch.stack([(pal_tree - torch.tensor(c, device=dev, dtype=torch.float32)).norm(dim=1).argmin() + 1 for c in U.BG_COLORS.values()])
bg_phrase = [tokenizer(f", on a {n} background", add_special_tokens=False)["input_ids"] for n in bg_names]
empty_ids = T16.encode_text(tokenizer, [""]).numpy()


def with_phrase(ids, phrase):
    """Insert phrase token ids before the first EOS of a padded CLIP id row (keeps length, EOS last if cut)."""
    row = list(ids)
    e = row.index(EOS)
    row = row[:e] + phrase + [EOS] * (len(row) - e)
    row = row[: len(ids)]
    if EOS not in row:
        row[-1] = EOS
    return row


def make_batch(n32_idx, n16_idx, rng):
    x = torch.cat([img32[torch.as_tensor(n32_idx, device=dev)].long(),
                   img16[torch.as_tensor(n16_idx, device=dev)].long().view(-1, 16, 16).repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(-1, U.N_PIX)])
    ids = np.concatenate([ids32[n32_idx], txt16[n16_idx]]).copy()
    size = torch.tensor([U.SIZE32] * len(n32_idx) + [U.SIZE16] * len(n16_idx), device=dev)
    bg = (torch.rand(len(ids), device=dev) < BG_PROB).long()
    for i in torch.nonzero(bg).flatten().tolist():
        c = rng.integers(len(bg_names))
        x[i][x[i] == 0] = bg_idx[c]
        if rng.random() < NAME_BG:
            ids[i] = with_phrase(ids[i], bg_phrase[c])
    drop = rng.random(len(ids)) < CFG_DROP
    ids[drop] = empty_ids
    i32, i16 = torch.as_tensor(n32_idx, device=dev), torch.as_tensor(n16_idx, device=dev)
    fill = torch.cat([fill_b32[i32], fill_b16[i16]])
    detail = torch.cat([detail_b32[i32], detail_b16[i16]])
    hide = torch.rand(len(ids), device=dev) < COND_DROP
    fill, detail = fill.masked_fill(hide, 0), detail.masked_fill(hide, 0)
    return x, torch.from_numpy(ids).long().to(dev), size, bg, fill, detail


# ---------------------------------------------------------------- model
model = U.UbaNova32().to(dev)
ckpt_path = f"{OUT}/ckpt32b.pt"
prev = sorted(glob.glob("/kaggle/input/**/ckpt32b.pt", recursive=True))
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler()
n32_b = round(BATCH * (1 - FRAC16))
per_epoch = len(tr32) // n32_b
total = EPOCHS * per_epoch
step, log = 0, []
src = ckpt_path if os.path.exists(ckpt_path) else (prev[0] if prev else None)
if src:
    ck = torch.load(src, map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    step, log = ck["step"], ck.get("log", [])
    print(f"resumed at step {step} from {src}", flush=True)
else:
    U.load_weights(model, torch.load(find("model32.pt"), map_location="cpu", weights_only=True)["model"])
    print("initialized from UbaNova-32 v1", flush=True)
print(f"steps/epoch {per_epoch}  total {total}", flush=True)


def save_ckpt():
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "log": log}, ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)


@torch.no_grad()
def eval_loss(idx, n=512):
    model.eval()
    idx = idx[:n]
    tot = 0.0
    for i in range(0, len(idx), 64):
        b = idx[i : i + 64]
        x = img32[torch.as_tensor(b, device=dev)].long()
        with torch.autocast("cuda", dtype=torch.float16):
            th = clip(input_ids=torch.from_numpy(ids32[b]).long().to(dev)).last_hidden_state
            z = torch.zeros(len(b), dtype=torch.long, device=dev)
            logits = model(th, z + U.SIZE32, z + U.BG_OFF, x[:, :-1], fill=z, detail=z)
        tot += F.cross_entropy(logits.float().transpose(1, 2), x, reduction="sum").item()
    model.train()
    return tot / (len(idx) * U.N_PIX)


@torch.no_grad()
def write_samples(tag):
    """Rows: any/any, detail low, detail high, fill low, fill high (same 8 prompts, same seed)."""
    model.eval()
    prompts = ["red potion", "iron sword", "a cat wearing a wizard hat", "golden crown with rubies",
               "green slime monster", "wooden treasure chest", "blue dragon", "pink cupcake"]
    rows = [(0, 0), (0, 1), (0, 3), (1, 0), (3, 0)]
    from PIL import Image
    sheet = Image.new("RGBA", (8 * 32, len(rows) * 32), (128, 128, 128, 255))
    with torch.autocast("cuda", dtype=torch.float16):
        th = clip(input_ids=T16.encode_text(tokenizer, prompts).long().to(dev)).last_hidden_state
        eh = clip(input_ids=torch.from_numpy(empty_ids).long().to(dev)).last_hidden_state
    zz = torch.zeros(8, dtype=torch.long, device=dev)
    for r, (fv, dv) in enumerate(rows):
        torch.manual_seed(11)
        with torch.autocast("cuda", dtype=torch.float16):
            img = U.sample(model, th, eh, zz + U.SIZE32, zz + U.BG_OFF, fill=zz + fv, detail=zz + dv).cpu().numpy()
        for i, t in enumerate(img):
            t = t.reshape(32, 32)
            a = np.concatenate([palette[t], ((t != 0) * 255).astype(np.uint8)[..., None]], -1).astype(np.uint8)
            sheet.alpha_composite(Image.fromarray(a), (i * 32, r * 32))
    sheet.resize((8 * 32 * 4, len(rows) * 32 * 4), Image.NEAREST).convert("RGB").save(f"{OUT}/samples_{tag}.png")
    model.train()


if step == 0:
    log.append({"step": 0, "val": eval_loss(ev32["val"]), "val_binding": eval_loss(ev32["val_binding"])})
    print(log[-1], flush=True)
    write_samples("init")

model.train()
rng = np.random.default_rng(step)
last_ckpt = t0 = time.time()
stopped_early = False
while step < total:
    epoch, j = divmod(step, per_epoch)
    perm = np.random.default_rng(1234 + epoch).permutation(tr32)
    for j in range(j, per_epoch):
        b32 = perm[j * n32_b : (j + 1) * n32_b]
        b16 = rng.integers(len(img16), size=BATCH - n32_b)
        x, ids, size, bg, fill, detail = make_batch(b32, b16, rng)
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * step / total))
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.no_grad():
                th = clip(input_ids=ids).last_hidden_state
            logits = model(th, size, bg, x[:, :-1], fill=fill, detail=detail)
        loss = F.cross_entropy(logits.float().transpose(1, 2), x)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % 200 == 0:
            print(f"step {step}/{total}  epoch {step / per_epoch:.2f}  loss {loss.item():.4f}  {(time.time() - t0) / 60:.0f} min", flush=True)
        if time.time() - last_ckpt > CKPT_S:
            save_ckpt(); last_ckpt = time.time()
        if time.time() - t_start > BUDGET_S:
            stopped_early = True
            break
    if stopped_early:
        break
    log.append({"step": step, "epoch": epoch + 1, "val": eval_loss(ev32["val"]), "val_binding": eval_loss(ev32["val_binding"]),
                "val_unseen": eval_loss(ev32["val_unseen"])})
    print(log[-1], flush=True)
    save_ckpt(); last_ckpt = time.time()
    write_samples(f"epoch{epoch + 1}")

save_ckpt()
json.dump(log, open(f"{OUT}/train_log.json", "w"), indent=1)
if step >= total:
    torch.save({"model": {k: v.half() for k, v in model.state_dict().items()}, "step": step, "version": "1.1",
                "fill_thresholds": FILL_Q.tolist(), "detail_thresholds": DETAIL_Q.tolist()}, f"{OUT}/model32_v11.pt")
    np.save(f"{OUT}/palette.npy", palette)
    print("training finished; model32_v11.pt written", flush=True)
else:
    print(f"stopped at step {step} (time budget); rerun with this output attached to resume", flush=True)
