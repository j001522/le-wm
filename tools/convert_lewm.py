"""Rename an HF-hosted LeWM checkpoint's encoder keys to the installed ViT naming,
so strict load_state_dict succeeds. Usage: python tools/convert_lewm.py quentinll/lewm-tworooms
"""

import re
import shutil
import sys
from pathlib import Path

import torch

name = sys.argv[1] if len(sys.argv) > 1 else "quentinll/lewm-pusht"
ckpt_dir = Path.home() / ".stable_worldmodel" / "checkpoints" / f"models--{name.replace('/', '--')}"
src = ckpt_dir / "weights.pt"
backup = ckpt_dir / "weights_hf_naming.pt.bak"

if not backup.exists():
    shutil.copy2(src, backup)
    print("backed up original to", backup.name)

sd = torch.load(backup, map_location="cpu", weights_only=True)

pairs = [
    (r"encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query", r"encoder.layers.\1.attention.q_proj"),
    (r"encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key", r"encoder.layers.\1.attention.k_proj"),
    (r"encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value", r"encoder.layers.\1.attention.v_proj"),
    (r"encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense", r"encoder.layers.\1.attention.o_proj"),
    (r"encoder\.encoder\.layer\.(\d+)\.intermediate\.dense", r"encoder.layers.\1.mlp.fc1"),
    (r"encoder\.encoder\.layer\.(\d+)\.output\.dense", r"encoder.layers.\1.mlp.fc2"),
    (r"encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1."),
]

out = {}
for k, v in sd.items():
    nk = k
    for pat, rep in pairs:
        new = re.sub(pat, rep, nk)
        if new != nk:
            nk = new
            break
    out[nk] = v

torch.save(out, src)
print("converted", len(out), "keys")
