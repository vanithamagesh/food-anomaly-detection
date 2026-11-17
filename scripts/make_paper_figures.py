"""Publication figures regenerated from the v6 result tables (600 dpi, consistent style)."""
import math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.ticker as tk

import argparse
HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser(description="Redraw the manuscript figures from the result tables.")
_ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
_ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
_args, _ = _ap.parse_known_args()
RES, OUT = _args.results, _args.out
os.makedirs(OUT, exist_ok=True)
for _f in ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"):
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 600, "mathtext.fontset": "stix"})
W2, W1 = 190 / 25.4, 90 / 25.4          # Elsevier double / single column width (inches)

ORDER = ["fusion_light", "fusion", "texture", "padim", "ae", "colour"]
NAME = {"fusion_light": "CT-R18 (proposed)", "fusion": "CT-WRN50", "texture": "PatchCore-WRN50",
        "padim": "PaDiM-R18", "ae": "CAE", "colour": "CIELAB-ΔE", "texture_light": "PatchCore-R18",
        "fusion_max": "CT-WRN50, fixed fusion"}
COL = {"fusion_light": "#C0392B", "fusion": "#E69F00", "texture": "#0072B2", "padim": "#009E73",
       "ae": "#8C8C8C", "colour": "#56B4E9", "texture_light": "#7FA7D1"}
FOODS = ["cashew", "chewinggum", "fryum", "pipe_fryum", "macaroni1", "macaroni2"]
FN = {"cashew": "Cashew", "chewinggum": "Chewing gum", "fryum": "Fryum", "pipe_fryum": "Pipe fryum",
      "macaroni1": "Macaroni 1", "macaroni2": "Macaroni 2"}


def label(ax, s, x=-0.09, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("png", "pdf", "tiff"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03, **kw)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Fig. 1 pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(W2, 3.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc, bold=False, fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
                                    fc=fc, ec="#333333", lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", shrinkA=1, shrinkB=1))

    A, B, C, D = "#EAF2F8", "#FDEDEC", "#FEF5E7", "#EAFAF1"
    ax.text(1, 42, "(a) Training — defect-free images only", fontsize=8, fontweight="bold", va="center")
    ax.text(1, 19.5, "(b) Inference and explanation — new product image", fontsize=8, fontweight="bold", va="center")
    # training row
    box(1, 27, 13, 11, "Defect-free\ntraining images\n(N = 405–810\nper product)", A)
    box(19, 32, 19, 7, "Pretrained CNN\nResNet-18 or WRN-50", A)
    box(19, 25, 19, 6, "sRGB → CIELAB\n(8 × 8-pixel cells)", A)
    box(43, 32, 18, 7, "Texture memory bank\n(greedy coreset)", A)
    box(43, 25, 18, 6, "ΔE look-up table\n(nearest normal colour)", A)
    box(66, 31.5, 15, 7.5, "Held-out defect-free\ncalibration images", C)
    box(66, 23.5, 15, 7, "Synthetic anomalies\n(no real defects)", C)
    box(85, 27, 14, 11, "z-normalisation,\nthreshold (5% FRR),\ncolour margin m", C)
    for y in (35.5, 28):
        arrow(14, 32.5, 19, y)
    arrow(38, 35.5, 43, 35.5); arrow(38, 28, 43, 28)
    arrow(61, 35.5, 66, 35.2); arrow(61, 28, 66, 27)
    arrow(81, 35.2, 85, 33.5); arrow(81, 27, 85, 30)
    # inference row
    box(1, 4, 13, 11, "Test image\n(unseen defects)", B)
    box(19, 9.5, 17, 6, "Texture anomaly\nmap  z_texture", B)
    box(19, 2.5, 17, 6, "Colour anomaly\nmap  z_colour", B)
    box(41, 4, 17, 10.5, "Fusion\nz = max(z_texture,\nz_colour − m)", B)
    box(63, 4, 13, 10.5, "Decision\nmax z ≥ threshold\n→ reject", B)
    box(81, 11, 18, 6, "Where: anomaly map", D, fs=6.8)
    box(81, 4.5, 18, 6, "Why: colour vs texture", D, fs=6.8)
    box(81, -2, 18, 6, "Compared with: nearest\nnormal patch, ΔE", D, fs=6.8)
    arrow(14, 9.5, 19, 12.5); arrow(14, 9.5, 19, 5.5)
    arrow(36, 12.5, 41, 10.5); arrow(36, 5.5, 41, 8)
    arrow(58, 9.25, 63, 9.25)
    for y in (14, 7.5, 1):
        arrow(76, 9.25, 81, y)
    ax.text(90.5, -4.4, "Explainable output", ha="center", fontsize=6.8, style="italic")
    arrow(92, 27, 70, 15.2)
    ax.set_ylim(-5.5, 44)
    save(fig, "Fig1_pipeline")


# ---------------------------------------------------------------- Fig. 3 detection by product
def fig_detection():
    ci = pd.read_csv(f"{RES}/results_with_CI95_v6.csv")
    ci = ci[ci.food.isin(FOODS)]
    fig, axes = plt.subplots(2, 1, figsize=(W2, 4.6), sharex=True)
    x = np.arange(len(FOODS)); w = 0.8 / len(ORDER)
    for k, (col, lo, hi, yl, lab) in enumerate((("image_AUROC", "AUROC_CI95_low", "AUROC_CI95_high", "Image-level AUROC (%)", "a"),
                                                  ("defects_caught", "caught_CI95_low", "caught_CI95_high", "Recall at operating threshold (%)", "b"))):
        ax = axes[k]
        for i, m in enumerate(ORDER):
            d = ci[ci.method == m].set_index("food").loc[FOODS]
            v = 100 * d[col].values
            err = np.vstack([v - 100 * d[lo].values, 100 * d[hi].values - v])
            ax.bar(x + (i - (len(ORDER) - 1) / 2) * w, v, w * 0.9, color=COL[m], label=NAME[m],
                   yerr=err, error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5, ecolor="#333333"))
        ax.set_ylabel(yl)
        ax.set_ylim(40 if k == 0 else 0, 100)
        ax.yaxis.grid(True, lw=0.3, color="#DDDDDD"); ax.set_axisbelow(True)
        label(ax, f"({lab})", x=-0.065)
    axes[1].set_xticks(x); axes[1].set_xticklabels([FN[f] for f in FOODS])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.53, 1.02), handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "Fig3_detection_by_product")


# ---------------------------------------------------------------- Fig. 5 confusion matrices
def fig_confusion():
    o = pd.read_csv(f"{RES}/confusion_metrics_v6.csv")
    L = o[o.method == "fusion_light"].reset_index(drop=True)
    names = dict(FN, ALL="All products")
    fig, axes = plt.subplots(1, 7, figsize=(W2, 1.75))
    for ax, (_, r) in zip(axes, L.iterrows()):
        M = np.array([[r.TN, r.FP], [r.FN, r.TP]]); P = M / M.sum(1, keepdims=True)
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(M[i, j])}\n({100 * P[i, j]:.0f}%)", ha="center", va="center", fontsize=6.3,
                        color="white" if P[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Defective"], fontsize=6.3)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Defective"] if ax is axes[0] else ["", ""], fontsize=6.3)
        ax.set_title(names[r.food], fontsize=7.5, pad=3); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4)
    axes[0].set_ylabel("Actual class", fontsize=7.5)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.25, wspace=0.14)
    fig.text(0.53, 0.105, "Predicted class (CT-R18)", ha="center", fontsize=7.5)
    save(fig, "Fig4_confusion_matrices")


# ---------------------------------------------------------------- Fig. 6 defect categories
def fig_categories():
    d = pd.read_csv(f"{RES}/defect_types_pooled_v6.csv")
    d = d[d.n_images >= 10].sort_values("caught_fusion")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]; y = np.arange(len(d))
    ax.barh(y + 0.19, 100 * d.caught_texture, 0.36, color=COL["texture"], label=NAME["texture"])
    ax.barh(y - 0.19, 100 * d.caught_fusion, 0.36, color=COL["fusion"], label=NAME["fusion"])
    ax.set_yticks(y); ax.set_yticklabels([f"{g} (n = {n})" for g, n in zip(d.group, d.n_images)])
    ax.set_xlim(0, 100); ax.set_xlabel("Recall at operating threshold (%)")
    ax.xaxis.grid(True, lw=0.3, color="#DDDDDD"); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, handlelength=1.0); label(ax, "(a)", x=-0.47)
    ax = axes[1]
    e = d.dropna(subset=["colour_share"]).sort_values("colour_share")
    y = np.arange(len(e)); c = 100 * e.colour_share.values
    ax.barh(y, c, 0.6, color="#2E86C1", label="Colour (CIELAB ΔE)")
    ax.barh(y, 100 - c, 0.6, left=c, color="#D35400", label="Texture (CNN features)")
    for yi, v in zip(y, c):
        ax.text(min(v + 1.5, 92), yi, f"{v:.0f}%", va="center", fontsize=6.5, color="white" if v > 92 else "black")
    ax.set_yticks(y); ax.set_yticklabels(e.group); ax.set_xlim(0, 100)
    ax.set_xlabel("Share of flagged area inside the defect (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, handlelength=1.0)
    label(ax, "(b)", x=-0.42)
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig6_defect_categories")


# ---------------------------------------------------------------- Fig. 7 data efficiency
def fig_data():
    g = pd.read_csv(f"{RES}/good_images_needed_v6.csv")
    g = g[g.method == "fusion"]
    pal = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    full = g.groupby("food").n_good.max()
    part = sorted(n for n in g.n_good.unique() if n not in set(full.values))
    xall = 2 * max(part)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    for ax, col, yl, lab in ((axes[0], "image_auroc", "Image-level AUROC (%)", "(a)"),
                             (axes[1], "defects_caught", "Recall at operating threshold (%)", "(b)")):
        for k, f in enumerate(FOODS):
            d = g[g.food == f].groupby("n_good")[col].agg(["mean", "std"]).reset_index()
            xs = [xall if n == full[f] else n for n in d.n_good]
            m, s = 100 * d["mean"].values, 100 * d["std"].fillna(0).values
            ax.plot(xs, m, "-o", ms=2.6, lw=1.0, color=pal[k], label=FN[f])
            ax.fill_between(xs, m - s, m + s, color=pal[k], alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.xaxis.set_minor_locator(tk.NullLocator())
        ax.set_xticks(part + [xall]); ax.set_xticklabels([str(p) for p in part] + ["All"])
        ax.set_xlabel("Number of defect-free training images"); ax.set_ylabel(yl)
        ax.set_ylim(0 if col == "defects_caught" else 50, 100)
        ax.yaxis.grid(True, lw=0.3, color="#DDDDDD"); ax.set_axisbelow(True)
        label(ax, lab, x=-0.16)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.4)
    fig.tight_layout()
    save(fig, "Fig7_data_efficiency")


# ---------------------------------------------------------------- Fig. 8 latency
def fig_latency():
    sp = pd.read_csv(f"{RES}/speed_v6.csv")
    ms = ["fusion_light", "texture_light", "fusion", "texture", "padim", "ae", "colour"]
    get = {(r.device, r.method): r.ms_per_image for r in sp.itertuples()}
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 2.9))
    y = np.arange(len(ms))
    for k, (dev, c, nm) in enumerate((("GPU", "#2C3E50", "GPU (NVIDIA T4)"), ("CPU", "#E67E22", "CPU only"))):
        v = [get[(dev, m)] for m in ms]; yy = y - 0.19 + k * 0.38
        ax.barh(yy, v, 0.34, color=c, label=nm)
        for yi, vi in zip(yy, v):
            ax.text(vi * 1.12, yi, f"{vi:.0f}", va="center", fontsize=6.3)
    ax.set_xscale("log"); ax.set_xlim(2, 3000)
    ax.set_xticks([5, 20, 100, 500]); ax.xaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(tk.NullLocator())
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in ms]); ax.invert_yaxis()
    ax.set_xlabel("Inference latency per image (ms, log scale)")
    ax.axvline(1000 / 15, ls=":", lw=0.6, color="#777777")
    ax.text(1000 / 15 * 1.06, len(ms) - 0.45, "15 images/s", fontsize=6, color="#555555")
    ax.legend(loc="lower right")
    ax.xaxis.grid(True, lw=0.3, color="#DDDDDD"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "Fig8_inference_latency")


if __name__ == "__main__":
    fig_pipeline(); fig_detection(); fig_confusion(); fig_categories(); fig_data(); fig_latency()
