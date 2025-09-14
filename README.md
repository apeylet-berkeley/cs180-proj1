# CS180/280A — Project 1  
**Colorizing the Prokudin-Gorskii Collection**  
Author: **Antoine Peylet**

This repository contains my baseline solution for re-colorizing early 1900s glass-plate photographs scanned as vertically stacked **B/G/R** grayscale channels. I split the plate into thirds and align **G** and **R** to **B** using **integer translations** only, scored with **NCC** (default) or **SSD**. Small JPGs run an exhaustive search; large TIFFs are handled with a **coarse→fine image pyramid** for speed and robustness.

## What I implemented
- Split the plate into equal thirds (**B, G, R**) and use **B** as the reference.
- Alignment by **integer (x, y) translations** only (no rotation/scale).
- Scoring metrics: **NCC** (maximize) and **SSD/L2** (minimize).
- **Border crop** (~10–12%) during scoring to avoid decorative frames skewing the metric.
- **Single-scale** exhaustive search for small images (default ±15 px window).
- **Pyramid** for large TIFFs: downsample ×0.5 until the min dimension ≲ 400 px, align coarsest → upsample shift → refine at each level.
- Outputs: aligned RGB `.jpg` files and a `results.json` with per-image **offsets** reported as `(x, y)` for G→B and R→B.

> This is the baseline only (no extra credit features).

## Repository layout
