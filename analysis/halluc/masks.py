"""Foreground masks for the masked hallucination metrics (colour rule per task, in HSV).

Same rule as dreamer4's dmc_background_mask.foreground_mask (AIC-Thesis/dreamer4), with a preset
per LeWM task:
    foreground = hue inside one of keep_hues AND saturation >= s_min AND value >= v_min,
then a small binary closing, hole filling, and specks smaller than min_area px dropped.

pusht: white background and grey border (S = 0). Foreground = the slate-blue T block (hue ~213,
S ~0.21, V ~0.71), the blue agent disc (hue ~226, S ~0.66) and the green goal T (hue ~122, S ~0.40).
The goal is static in an episode, so it only counts when the imagination moves, loses or invents it.
"""

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage

PRESETS = {
    "pusht": dict(keep_hues=((190.0, 250.0), (95.0, 150.0)), s_min=0.10, v_min=0.30, close=1, min_area=6),
}


def preset_for_task(task):
    t = str(task)
    for name in PRESETS:
        if t == name or t.startswith(name + "-"):
            return name
    raise KeyError(f"no foreground-mask preset for task {task!r} (add one to analysis/halluc/masks.py)")


def _in_band(h, lo, hi):
    return (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)  # wraps through 0


def foreground_mask(frame, preset):
    """(H,W,3) uint8 or float in [0,1] -> (H,W) bool, True = foreground."""
    cfg = PRESETS[preset]
    f = np.asarray(frame)
    f = f.astype(np.float32) / 255.0 if f.dtype == np.uint8 else f.astype(np.float32)
    hsv = rgb_to_hsv(np.clip(f, 0.0, 1.0))
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    hue_ok = np.zeros(h.shape, bool)
    for lo, hi in cfg["keep_hues"]:
        hue_ok |= _in_band(h, lo, hi)
    m = hue_ok & (s >= cfg["s_min"]) & (v >= cfg["v_min"])
    if cfg["close"] > 0 and m.any():
        m = ndimage.binary_closing(m, iterations=cfg["close"], border_value=0)
    m = ndimage.binary_fill_holes(m)
    if cfg["min_area"] > 1 and m.any():
        lab, n = ndimage.label(m)
        sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= cfg["min_area"]
        m = keep[lab]
    return m
