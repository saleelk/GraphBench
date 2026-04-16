#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Saleel Kudchadker

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

CSV_FILE = os.path.join(os.path.dirname(__file__), "results.csv")
OUT_DIR  = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = {}  # data[version][topology] = {size: us}

with open(CSV_FILE) as f:
    for row in csv.DictReader(f):
        if row["mode"] != "no-sync":
            continue
        ver  = row["rocm_version"]
        size = int(row["size"])
        for topo in ("straight", "paths2", "full2"):
            data.setdefault(ver, {}).setdefault(topo, {})[size] = float(row[topo])

versions = sorted(data.keys())
sizes    = sorted(next(iter(next(iter(data.values())).values())).keys())

VERSION_LABELS = {
    "7.2":  "ROCm 7.2",
    "7.12": "ROCm 7.12",
    "7.13": "ROCm 7.13",
}
COLORS = {
    "7.2":  "#e05c5c",
    "7.12": "#f0a030",
    "7.13": "#4caf7d",
}
MARKERS = {
    "7.2":  "o",
    "7.12": "s",
    "7.13": "^",
}

TOPO_TITLES = {
    "straight": "straight  —  1 linear chain of N nodes",
    "paths2":   "paths2  —  lead → 2 parallel branches → tail",
    "full2":    "full2  —  2 fully independent chains of N/2 nodes",
}

# ---------------------------------------------------------------------------
# One plot per topology
# ---------------------------------------------------------------------------
for topo, title in TOPO_TITLES.items():
    fig, ax = plt.subplots(figsize=(9, 5))

    for ver in versions:
        ys = [data[ver][topo][s] for s in sizes]
        ax.plot(
            sizes, ys,
            label=VERSION_LABELS.get(ver, ver),
            color=COLORS.get(ver, None),
            marker=MARKERS.get(ver, "o"),
            linewidth=2,
            markersize=5,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Graph size (nodes)", fontsize=12)
    ax.set_ylabel("hipGraphLaunch latency (µs)", fontsize=12)
    ax.set_title(f"{title}\n(no-sync: submission latency only)", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xticks(sizes)
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"plot_{topo}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
