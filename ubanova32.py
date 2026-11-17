"""UbaNova-32: text -> 32x32 pixel-art sprite, with size and background control.

Sequence: [CLIP text states (32) | SIZE | BG | BOS | 1024 pixels], length 1059.
  SIZE: 0 = "16x16 look" (trained on 16x16 sprites upscaled 2x; output can be downsampled to 16x16), 1 = native 32x32
  BG:   0 = transparent background, 1 = background filled with a color (named in the prompt or chosen by the model)
Pixels are indices into the UbaNova palette (4097 entries, 0 = transparent), same as UbaNova-16.
Weights are initialized from UbaNova-16 (see init_from_16) and fine-tuned.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 4097
TEXT_LEN = 32
SIDE = 32
N_PIX = SIDE * SIDE
N_COND = 2
PREFIX = TEXT_LEN + N_COND  # BOS sits at this position
D, LAYERS, HEADS = 512, 12, 8

SIZE16, SIZE32 = 0, 1
BG_OFF, BG_ON = 0, 1
# Optional style controls (v1.1). 0 = unspecified (the model behaves like v1); 1..3 = low / medium / high.
# FILL: how much of the canvas the object covers. DETAIL: how many colors the sprite uses.
COND_ANY, COND_LOW, COND_MID, COND_HIGH = 0, 1, 2, 3

# Background colors used for training augmentation and accepted in prompts ("on a <name> background").
BG_COLORS = {
    "black": (20, 20, 24), "white": (240, 240, 240), "gray": (128, 128, 128), "red": (200, 40, 40),
    "orange": (230, 130, 40), "yellow": (240, 210, 60), "green": (60, 170, 70), "dark green": (30, 90, 40),
    "blue": (50, 90, 210), "light blue": (130, 190, 240), "navy": (25, 35, 90), "purple": (130, 70, 180),
    "pink": (240, 140, 190), "brown": (120, 80, 50), "beige": (225, 205, 165), "teal": (40, 150, 150),
}


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class UbaNova32(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(512, D)
        self.tok = nn.Embedding(VOCAB + 1, D)  # last id is BOS
        self.size_emb = nn.Embedding(2, D)
        self.bg_emb = nn.Embedding(2, D)
        self.fill_emb = nn.Embedding(4, D)  # added to the SIZE slot, so the sequence layout is unchanged
        self.detail_emb = nn.Embedding(4, D)
        nn.init.zeros_(self.fill_emb.weight)  # zero = no effect: a v1 checkpoint behaves exactly as before
        nn.init.zeros_(self.detail_emb.weight)
        self.pos = nn.Parameter(torch.randn(1, PREFIX + 1 + N_PIX, D) * 0.02)
        self.blocks = nn.ModuleList(Block(D, HEADS) for _ in range(LAYERS))
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB)

    def prefix(self, text_h, size, bg, fill=None, detail=None):
        b = text_h.shape[0]
        bos = torch.full((b, 1), VOCAB, dtype=torch.long, device=text_h.device)
        cond = self.size_emb(size)
        if fill is not None:
            cond = cond + self.fill_emb(fill)
        if detail is not None:
            cond = cond + self.detail_emb(detail)
        return torch.cat([self.text_proj(text_h), cond[:, None], self.bg_emb(bg)[:, None], self.tok(bos)], 1)

    def forward(self, text_h, size, bg, prev, fill=None, detail=None):
        """prev: (B, t) pixels so far -> logits (B, t + 1, VOCAB)."""
        x = torch.cat([self.prefix(text_h, size, bg, fill, detail), self.tok(prev)], 1)
        x = x + self.pos[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x[:, PREFIX:]))


@torch.no_grad()
def load_weights(model, sd):
    """Load a v1 (no fill/detail) or v1.1 state dict."""
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected and set(missing) <= {"fill_emb.weight", "detail_emb.weight"}, (missing, unexpected)


def init_from_16(model, sd16):
    """Copy UbaNova-16 weights; interpolate its 16x16 pixel position grid to 32x32."""
    own = model.state_dict()
    for k, v in sd16.items():
        if k != "pos" and k in own and own[k].shape == v.shape:
            own[k].copy_(v.float())
    pos16 = sd16["pos"].float()  # (1, 32 + 1 + 256, D): text | BOS | pixels
    own["pos"][:, :TEXT_LEN] = pos16[:, :TEXT_LEN]
    own["pos"][:, PREFIX] = pos16[:, TEXT_LEN]
    grid = pos16[:, TEXT_LEN + 1 :].reshape(1, 16, 16, D).permute(0, 3, 1, 2)
    own["pos"][:, PREFIX + 1 :] = F.interpolate(grid, size=(SIDE, SIDE), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(1, N_PIX, D)
    model.load_state_dict(own)


def _attend(block, x, kv=None):
    b, t, c = x.shape
    q, k, v = block.qkv(block.ln1(x)).view(b, t, 3, block.heads, c // block.heads).permute(2, 0, 3, 1, 4)
    if kv is not None:
        k, v = torch.cat([kv[0], k], 2), torch.cat([kv[1], v], 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=kv is None)
    x = x + block.proj(y.transpose(1, 2).reshape(b, t, c))
    return x + block.mlp(block.ln2(x)), (k, v)


def _pick(logits, temperature, top_p):
    probs = F.softmax(logits / max(temperature, 1e-4), -1)
    if top_p < 1.0:  # nucleus: drop rare palette entries that show up as stray pixels
        sp, si = probs.sort(-1, descending=True)
        sp[(sp.cumsum(-1) - sp) > top_p] = 0
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return torch.multinomial(probs, 1)


@torch.no_grad()
def sample(model, text_h, empty_h, size, bg, cfg=3.0, temperature=0.9, top_p=0.9, fill=None, detail=None):
    """KV-cached sampling with classifier-free guidance. size/bg: (B,) long. Returns (B, 1024) palette indices.
    With bg == BG_OFF the transparent index is the only background the model has learned for that flag."""
    b = text_h.shape[0]
    th = torch.cat([text_h, empty_h.expand(b, -1, -1)])
    size2, bg2 = torch.cat([size, size]), torch.cat([bg, bg])
    fill2 = torch.cat([fill, fill]) if fill is not None else None
    detail2 = torch.cat([detail, detail]) if detail is not None else None
    x = model.prefix(th, size2, bg2, fill2, detail2) + model.pos[:, : PREFIX + 1]
    kv = []
    for block in model.blocks:
        x, c = _attend(block, x)
        kv.append(c)
    logits = model.head(model.ln_f(x[:, -1])).float()
    out = []
    for i in range(N_PIX):
        if i:
            x = model.tok(torch.cat([out[-1], out[-1]])) + model.pos[:, PREFIX + i : PREFIX + i + 1]
            for j, block in enumerate(model.blocks):
                x, kv[j] = _attend(block, x, kv[j])
            logits = model.head(model.ln_f(x[:, 0])).float()
        cond, unc = logits.chunk(2)
        out.append(_pick(unc + cfg * (cond - unc), temperature, top_p))
    return torch.cat(out, 1)


def to_16(tokens):
    """(B, 1024) generated with SIZE16 -> (B, 256) by taking the top-left pixel of every 2x2 block."""
    return tokens.view(-1, SIDE, SIDE)[:, ::2, ::2].reshape(-1, 256)
