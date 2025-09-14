#!/usr/bin/env python3
"""
CS180/280A Project 1 — Colorizing the Prokudin‑Gorskii Collection

"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from skimage import io, img_as_float32, transform, util

# ---------------------------- Utils ----------------------------

def load_plate(path: str) -> np.ndarray:
    """Read image as float32 in [0,1]. Handles 8-bit JPG and 16-bit TIFF."""
    im = io.imread(path)
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    elif im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    else:
        im = img_as_float32(im)
    if im.ndim == 3:
        im = im[..., 0].astype(np.float32)
    return im


def split_bgr(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a vertically stacked plate, return (b, g, r)."""
    H = im.shape[0] // 3
    im = im[: 3 * H, :] 
    b = im[0:H, :]
    g = im[H:2*H, :]
    r = im[2*H:3*H, :]
    return b, g, r


def crop_interior(im: np.ndarray, margin_frac: float = 0.1) -> np.ndarray:
    """Crop a border margin for scoring; returns a view (no copy)."""
    h, w = im.shape[:2]
    mh = int(round(margin_frac * h))
    mw = int(round(margin_frac * w))
    return im[mh:h-mh, mw:w-mw] if (h-2*mh > 0 and w-2*mw > 0) else im


def shift_image(im: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift using np.roll (wraparound). We'll crop later for display."""
    return np.roll(np.roll(im, dy, axis=0), dx, axis=1)


# -------------------------- Metrics ---------------------------

def ssd(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of squared differences (lower is better)."""
    d = a - b
    return float(np.sum(d * d))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation (higher is better)."""
    av = a - a.mean()
    bv = b - b.mean()
    denom = (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-8)
    return float(np.sum(av * bv) / denom)


# ------------------------ Alignment Core ----------------------

def align_single_scale(moving: np.ndarray, ref: np.ndarray, win: int = 15,
                       metric: str = "ncc", margin_frac: float = 0.1) -> Tuple[int, int, float]:
    """
    Exhaustive search over integer shifts in [-win, +win] for dy, dx.
    Returns (best_dy, best_dx, best_score).
    """
    m = crop_interior(moving, margin_frac)
    r = crop_interior(ref, margin_frac)

    best_dy, best_dx = 0, 0
    if metric == "ssd":
        best_score = float("inf")
        better = lambda s, best: s < best
    else:
        metric = "ncc"
        best_score = -float("inf")
        better = lambda s, best: s > best

    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            ms = shift_image(m, dy, dx)
            if metric == "ssd":
                score = ssd(ms, r)
            else:
                score = ncc(ms, r)
            if better(score, best_score):
                best_score, best_dy, best_dx = score, dy, dx
    return best_dy, best_dx, best_score


def downsample(im: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Simple downsample using skimage.transform.resize (anti-aliased)."""
    h, w = im.shape[:2]
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    return transform.resize(im, (nh, nw), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)


def align_pyramid(moving: np.ndarray, ref: np.ndarray, base_win: int = 15, levels: int = None,
                  metric: str = "ncc", margin_frac: float = 0.1) -> Tuple[int, int]:
    """
    Coarse-to-fine alignment. If levels is None/"auto", compute until min dim < ~400.
    Returns (total_dy, total_dx).
    """
    if levels in (None, "auto"):
        levels = 0
        h, w = moving.shape[:2]
        while min(h, w) > 400:
            h, w = int(h * 0.5), int(w * 0.5)
            levels += 1

    mov_pyr = [moving.astype(np.float32)]
    ref_pyr = [ref.astype(np.float32)]
    for _ in range(levels):
        mov_pyr.append(downsample(mov_pyr[-1], 0.5))
        ref_pyr.append(downsample(ref_pyr[-1], 0.5))

    total_dy, total_dx = 0, 0
    for lvl in range(len(mov_pyr) - 1, -1, -1):
        m = mov_pyr[lvl]
        r = ref_pyr[lvl]

        win = max(5, int(base_win * (0.75 if lvl == 0 else 1.0)))

        best_dy, best_dx, best_score = 0, 0, -1e18
        if metric == "ssd":
            best_score = float("inf")

        for ddy in range(-win, win + 1):
            for ddx in range(-win, win + 1):
                dy = total_dy + ddy
                dx = total_dx + ddx
                ms = shift_image(m, dy, dx)
                if metric == "ssd":
                    score = ssd(crop_interior(ms, margin_frac), crop_interior(r, margin_frac))
                    better = score < best_score
                else:
                    score = ncc(crop_interior(ms, margin_frac), crop_interior(r, margin_frac))
                    better = score > best_score
                if better:
                    best_score, best_dy, best_dx = score, dy, dx

        total_dy, total_dx = best_dy, best_dx

        if lvl > 0:
            total_dy *= 2
            total_dx *= 2

    return int(total_dy), int(total_dx)


def stack_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack to RGB image, clipping to [0,1]."""
    rgb = np.dstack([r, g, b]).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


def compose_and_align(plate: np.ndarray, metric: str = "ncc",
                      pyramid: bool = True, base_win: int = 15, levels: str = "auto",
                      margin_frac: float = 0.1) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Split BGR, align G and R to B, return RGB image and offsets dict."""
    b, g, r = split_bgr(plate)

    if pyramid:
        dy_g, dx_g = align_pyramid(g, b, base_win=base_win, levels=levels, metric=metric, margin_frac=margin_frac)
        dy_r, dx_r = align_pyramid(r, b, base_win=base_win, levels=levels, metric=metric, margin_frac=margin_frac)
    else:
        dy_g, dx_g, _ = align_single_scale(g, b, win=base_win, metric=metric, margin_frac=margin_frac)
        dy_r, dx_r, _ = align_single_scale(r, b, win=base_win, metric=metric, margin_frac=margin_frac)

    g_aligned = shift_image(g, dy_g, dx_g)
    r_aligned = shift_image(r, dy_r, dx_r)
    rgb = stack_rgb(r_aligned, g_aligned, b)

    offsets = {"green": [int(dx_g), int(dy_g)], "red": [int(dx_r), int(dy_r)]} 
    return rgb, offsets


def save_image(path: str, rgb: np.ndarray) -> None:
    """Save as 8-bit JPG/PNG depending on extension."""
    out = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    io.imsave(path, out)


# ----------------------------- CLI ----------------------------

def process_one(in_path: Path, out_path: Path, args) -> Dict:
    plate = load_plate(str(in_path))
    rgb, offsets = compose_and_align(
        plate,
        metric=args.metric,
        pyramid=not args.no_pyramid,
        base_win=args.window,
        levels=("auto" if args.levels == "auto" else int(args.levels)),
        margin_frac=args.margin,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(str(out_path), rgb)
    return {
        "input": str(in_path.name),
        "output": str(out_path.name),
        "metric": args.metric,
        "offsets": offsets,
    }


def main():
    p = argparse.ArgumentParser(description="Colorize Prokudin-Gorskii plates.")
    p.add_argument("input", help="Path to a single image OR a folder containing images.")
    p.add_argument("output", help="Output file (if single image) OR output folder (if input is a folder).")
    p.add_argument("--metric", choices=["ncc", "ssd"], default="ncc", help="Scoring metric.")
    p.add_argument("--window", type=int, default=15, help="Search window half-size for single-scale/refinement.")
    p.add_argument("--levels", default="auto", help="Number of pyramid levels (int) or 'auto'.")
    p.add_argument("--margin", type=float, default=0.1, help="Fractional border ignored when scoring (0..0.3).")
    p.add_argument("--no-pyramid", action="store_true", help="Disable pyramid (use single-scale only).")
    p.add_argument("--ext", default=".jpg", help="Output extension when processing a folder (.jpg or .png).")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    results: List[Dict] = []

    if in_path.is_file():
        if out_path.is_dir():
            out_file = out_path / (in_path.stem + ".jpg")
        else:
            out_file = out_path
        res = process_one(in_path, out_file, args)
        results.append(res)
        print(json.dumps(res, indent=2))
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
        for f in sorted(in_path.iterdir()):
            if f.suffix.lower() in exts:
                out_file = out_path / (f.stem + args.ext)
                res = process_one(f, out_file, args)
                results.append(res)
                print(f"Saved → {out_file}  (offsets: {res['offsets']})")

    with open(out_path / "results.json" if out_path.is_dir() else Path(str(out_path) + "_results.json"), "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
