"""UbaNova-32 generator for the CLI and the web UI (CPU or GPU)."""
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

import ubanova32 as U

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = "openai/clip-vit-base-patch32"
HF_REPO = "ubanova/UbaNova-32"


def _resolve(path: Optional[str], *names: str) -> str:
    """Explicit path, else weights/<name> next to this file, else download the first name from the Hugging Face Hub."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    for name in names:
        local = os.path.join(HERE, "weights", name)
        if os.path.exists(local):
            return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, names[0])


class SpriteGenerator32:
    def __init__(self, model_path: Optional[str] = None, palette_path: Optional[str] = None, device: Optional[str] = None):
        from transformers import CLIPTextModel, CLIPTokenizerFast

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_fp16 = self.device.type == "cuda"
        self.model_path = _resolve(model_path, "model32_v11.pt", "model32.pt")  # prefer v1.1 (fill/detail controls)
        self.palette = np.load(_resolve(palette_path, "palette.npy"))
        self.tokenizer = CLIPTokenizerFast.from_pretrained(CLIP)
        self.clip = CLIPTextModel.from_pretrained(CLIP, torch_dtype=torch.float16 if self.use_fp16 else torch.float32).to(self.device).eval()
        self.model = U.UbaNova32()
        ck = torch.load(self.model_path, map_location="cpu", weights_only=True)
        U.load_weights(self.model, ck["model"])
        self.has_controls = ck.get("version") == "1.1"
        self.model.to(self.device).eval()

    def _cond(self, mode, n):
        if not self.has_controls:
            return None
        if mode == "auto":
            return torch.randint(1, 4, (n,), device=self.device)
        value = {"any": U.COND_ANY, "low": U.COND_LOW, "mid": U.COND_MID, "high": U.COND_HIGH}[mode]
        return torch.full((n,), value, dtype=torch.long, device=self.device)

    def _encode(self, captions):
        ids = self.tokenizer(captions, padding="max_length", max_length=U.TEXT_LEN, truncation=True, return_tensors="pt")["input_ids"]
        return self.clip(input_ids=ids.to(self.device)).last_hidden_state

    @torch.no_grad()
    def generate(self, prompt: str, n: int = 4, cfg: float = 2.0, temperature: float = 1.0, top_p: float = 0.9,
                 seed: Optional[int] = None, size: int = 32, background: bool = False,
                 fill: str = "auto", detail: str = "auto") -> List[Image.Image]:
        """size 32 = native 32x32; size 16 = chunky 16x16 look (returned as 16x16).
        background=True fills the background; name a color in the prompt ("..., on a blue background") to choose it.
        fill / detail (v1.1 models): "auto" (random, like the training data), "any", "low", "mid" or "high"."""
        if seed is not None:
            torch.manual_seed(seed)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_fp16):
            th, eh = self._encode([prompt] * n), self._encode([""])
            sz = torch.full((n,), U.SIZE16 if size == 16 else U.SIZE32, dtype=torch.long, device=self.device)
            bg = torch.full((n,), U.BG_ON if background else U.BG_OFF, dtype=torch.long, device=self.device)
            out = U.sample(self.model, th, eh, sz, bg, cfg, temperature, top_p,
                           fill=self._cond(fill, n), detail=self._cond(detail, n))
        if size == 16:
            out = U.to_16(out)
        side = 16 if size == 16 else 32
        images = []
        for t in out.cpu().numpy().reshape(-1, side, side):
            rgba = np.concatenate([self.palette[t], ((t != 0) * 255).astype(np.uint8)[..., None]], -1).astype(np.uint8)
            images.append(Image.fromarray(rgba, "RGBA"))
        return images
