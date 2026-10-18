"""Fig. 2. Usage: python fig2_framework.py OUT_DIR (run fig3_fig9.py first; it writes OUT_DIR/frames)"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image

OUT = sys.argv[1]
FR = os.path.join(OUT, "frames")
MM = 1 / 25.4
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "mathtext.bf": "Liberation Sans:bold", "mathtext.cal": "DejaVu Serif:italic"})
C = dict(E1="#000000", X="#888888", W="#009E73", E2="#CC79A7", pos="#D55E00", neg="#56B4E9", ok="#4D4D4D", bad="#FFFFFF",
         ink="#222222", soft="#F3F5F8", edge="#C9D1DC")


def save(fig, name):
    for ext in ("pdf", "png", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03, dpi=300 if ext == "png" else 600, **kw)
    plt.close(fig)


def letter(fig, x, y, L):
    fig.text(x, y, f"({L})", fontsize=8.5, fontweight="bold", ha="left", va="top")


fig = plt.figure(figsize=(190 * MM, 118 * MM))
letter(fig, 0.005, 0.995, "a")
mods = [("1", "Input", "#E8F1FA"), ("2", "Pair partition", "#E8F6F1"), ("3", "Eligibility and weights", "#FDF3E3"),
        ("4", "Unit-level inference", "#F6ECF3"), ("5", "Audit report", "#EFEFEF")]
x0, wbox, gap, yb, hb = 0.02, 0.172, 0.024, 0.535, 0.43
boxes = []
for k, (num, name, col) in enumerate(mods):
    xb = x0 + k * (wbox + gap)
    fig.patches.append(FancyBboxPatch((xb, yb), wbox, hb, boxstyle="round,pad=0.003,rounding_size=0.012",
                                      transform=fig.transFigure, fc=col, ec="#9AA5B4", lw=0.6, zorder=-10))
    fig.patches.append(FancyBboxPatch((xb, yb + hb - 0.055), wbox, 0.055, boxstyle="round,pad=0.003,rounding_size=0.012",
                                      transform=fig.transFigure, fc="#2F3B4C", ec="none", zorder=-9))
    fig.text(xb + 0.01, yb + hb - 0.028, f"{num}  {name}", color="white", fontsize=7, fontweight="bold", va="center")
    boxes.append(xb)
    if k < 4:
        fig.patches.append(FancyArrowPatch((xb + wbox + 0.002, yb + hb / 2), (xb + wbox + gap - 0.002, yb + hb / 2),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color="#2F3B4C", lw=1.0))
# module 1: unit cards with thumbnails
xb = boxes[0]
thumbs = [("196_35025", "unit 1 (op. 196)", [1, 1, 0, 0, 0, 0], "mixed-label"), ("192_23825", "unit 2 (op. 192)", [1, 1, 1, 1, 1, 1], "single-label")]
for i, (t, name, lab, kind) in enumerate(thumbs):
    yy = yb + hb - 0.175 - i * 0.135
    ax = fig.add_axes([xb + 0.008, yy, 0.062, 0.095]); ax.imshow(Image.open(f"{FR}/{t}.png")); ax.axis("off")
    fig.text(xb + 0.076, yy + 0.078, name, fontsize=6, fontweight="bold")
    for j, l in enumerate(lab):
        fig.patches.append(Rectangle((xb + 0.076 + j * 0.0145, yy + 0.04), 0.012, 0.022, transform=fig.transFigure,
                                     fc=C["pos"] if l else C["neg"], ec="white", lw=0.3))
    fig.text(xb + 0.076, yy + 0.012, kind, fontsize=6, color="#444444", style="italic")
fig.text(xb + 0.01, yb + 0.075, "labels $y_{ui}$ and scores $s_{ui}$\nfor every frame", fontsize=6, color="#333333")
fig.text(xb + 0.01, yb + 0.012, "frames nested in operations\nor clips (unit $u$)", fontsize=6, color="#333333")
# module 2: matrix
xb = boxes[1]
ax = fig.add_axes([xb + 0.02, yb + 0.17, 0.13, 0.17])
units = [(3, 5), (1, 6), (4, 2), (0, 5)]
pu = [u for u, (p, q) in enumerate(units) for _ in range(p)]; nu = [u for u, (p, q) in enumerate(units) for _ in range(q)]
for i, a in enumerate(pu):
    for j, b in enumerate(nu):
        ax.add_patch(Rectangle((j, len(pu) - 1 - i), 1, 1, fc=C["W"] if a == b else "#D5D9DF", ec="white", lw=0.4))
ax.set_xlim(0, len(nu)); ax.set_ylim(0, len(pu)); ax.axis("off")
fig.text(xb + 0.01, yb + 0.125, "all $N^+N^-$ pairs → within / cross", fontsize=6)
fig.text(xb + 0.01, yb + 0.085, "$\\alpha$ = within-unit fraction", fontsize=6)
fig.text(xb + 0.01, yb + 0.03, "$E_1 = \\alpha W + (1-\\alpha) X$", fontsize=6.6, color=C["ink"])
# module 3: weights bars
xb = boxes[2]
ax = fig.add_axes([xb + 0.02, yb + 0.2, 0.135, 0.14])
w = np.array([15, 8, 6, 0]); cols = [C["W"]] * 3 + ["#BBBBBB"]
ax.bar(range(4), w, color=cols, width=0.7)
for i, v in enumerate(w):
    ax.text(i, 0.6 if v == 0 else v - 0.6, "single" if v == 0 else f"{v}", ha="center", va="bottom" if v == 0 else "top", fontsize=6,
            color="#555555" if v == 0 else "white")
ax.axhline(np.mean(w[:3]), color=C["E2"], lw=0.8, ls="--")
ax.text(3.45, np.mean(w[:3]) + 0.6, "equal", fontsize=6, color=C["E2"], ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(["U1", "U3", "U2", "U4"], fontsize=6)
ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_ylim(0, 18)
fig.text(xb + 0.01, yb + 0.125, "weights $w_u = n_u^+ n_u^-$ (bars),", fontsize=6)
fig.text(xb + 0.01, yb + 0.085, "mixed-label set $\\mathcal{U}^*$, $n_{\\mathrm{eff}}$", fontsize=6)
fig.text(xb + 0.01, yb + 0.03, "$W$ (pair-weighted)  vs  $E_2$ (equal)", fontsize=6.0)
# module 4: bootstrap
xb = boxes[3]
orig = ["U1", "U2", "U3", "U4"]; draw = ["U3", "U1", "U3", "U4"]
for row, (lab, units_) in enumerate((("test set", orig), ("resample", draw))):
    yy = yb + hb - 0.105 - row * 0.05
    fig.text(xb + 0.01, yy + 0.008, lab, fontsize=6, va="center", color="#333333")
    for c, u in enumerate(units_):
        fig.patches.append(Rectangle((xb + 0.072 + c * 0.024, yy - 0.004), 0.02, 0.024, transform=fig.transFigure,
                                     fc="white", ec="#555555", lw=0.6, zorder=-5))
        fig.text(xb + 0.082 + c * 0.024, yy + 0.008, u, fontsize=6, ha="center", va="center")
axh = fig.add_axes([xb + 0.02, yb + 0.15, 0.135, 0.08])
rng = np.random.default_rng(3); d = rng.normal(0.04, 0.025, 4000)
axh.hist(d, bins=40, color="#BBBBBB", edgecolor="none")
lo, hi = np.quantile(d, [0.025, 0.975])
axh.axvline(lo, color="#333333", lw=0.7); axh.axvline(hi, color="#333333", lw=0.7)
axh.text(lo, axh.get_ylim()[1] * 1.02, "2.5%", ha="center", va="bottom", fontsize=6)
axh.text(hi, axh.get_ylim()[1] * 1.02, "97.5%", ha="center", va="bottom", fontsize=6)
axh.set_yticks([]); axh.set_xticks([]); axh.spines["left"].set_visible(False)
fig.text(xb + 0.01, yb + 0.115, "percentile CI; $t$ interval for $W$", fontsize=6)
fig.text(xb + 0.01, yb + 0.075, "matched across models", fontsize=6)
fig.text(xb + 0.01, yb + 0.035, "paired contrasts, BH-adjusted $q$", fontsize=6)
# module 5: report table (items of Table 2)
xb = boxes[4]
rows = ["comparison and estimand", "clustering unit", "$\\alpha$", "mixed-label units", "$n_{\\mathrm{eff}}$ of $W$",
        "estimands with unit-level CI", "contrasts", "paired model comparisons", "reference standard", "check of $E_1=\\alpha W+(1-\\alpha)X$", "$\\Delta W$ vs time baseline"]
for i, t in enumerate(rows):
    yy = yb + hb - 0.082 - i * 0.0285
    fig.patches.append(Rectangle((xb + 0.008, yy - 0.012), wbox - 0.016, 0.026, transform=fig.transFigure,
                                 fc="white" if i % 2 == 0 else "#F4F4F4", ec="#DDDDDD", lw=0.3, zorder=-5))
    fig.text(xb + 0.013, yy + 0.001, f"{i+1:>2}  {t}", fontsize=6, va="center")
fig.text(xb + 0.01, yb + 0.012, "items of Table 2", fontsize=6, color="#333333")
# bottom: toy example (b)(c)(d)
units = [("Unit 1", 3, 5), ("Unit 2", 1, 6), ("Unit 3", 4, 2), ("Unit 4", 0, 5)]
letter(fig, 0.005, 0.49, "b")
ax = fig.add_axes([0.02, 0.03, 0.27, 0.43]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(-0.7, 8.2)
for k, (name, npos, nneg) in enumerate(units):
    y0 = 7.0 - k * 2.0
    ax.add_patch(FancyBboxPatch((0.1, y0 - 0.55), 9.6, 1.45, boxstyle="round,pad=0.02,rounding_size=0.25", fc="#F4F4F4", ec="#CCCCCC", lw=0.5))
    ax.text(0.4, y0 + 0.6, name, fontsize=6.5, va="center")
    for j, l in enumerate([1] * npos + [0] * nneg):
        ax.add_patch(Rectangle((0.4 + j * 1.05, y0 - 0.38), 0.85, 0.62, fc=C["pos"] if l else C["neg"], ec="white", lw=0.4))
    ax.text(9.45, y0 + 0.6, "mixed-label" if npos and nneg else "single-label", fontsize=6, ha="right", va="center",
            color="#333333" if npos and nneg else "#888888", style="italic")
ax.add_patch(Rectangle((0.4, -0.62), 0.55, 0.4, fc=C["pos"], ec="none")); ax.text(1.15, -0.42, "positive frame", fontsize=6, va="center")
ax.add_patch(Rectangle((5.2, -0.62), 0.55, 0.4, fc=C["neg"], ec="none")); ax.text(5.95, -0.42, "negative frame", fontsize=6, va="center")
letter(fig, 0.33, 0.445, "c")
ax = fig.add_axes([0.37, 0.09, 0.27, 0.33])
for i, a in enumerate(pu):
    for j, b in enumerate(nu):
        ax.add_patch(Rectangle((j, len(pu) - 1 - i), 1, 1, fc=C["W"] if a == b else "#DDDDDD", ec="white", lw=0.5))
ax.set_xlim(0, len(nu)); ax.set_ylim(0, len(pu)); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
for xb_ in np.cumsum([q for _, p_, q in units])[:-1]: ax.axvline(xb_, color="#333333", lw=0.8)
for yb_ in len(pu) - np.cumsum([p_ for _, p_, q in units if p_])[:-1]: ax.axhline(yb_, color="#333333", lw=0.8)
ax.set_xlabel(f"{len(nu)} negative frames, grouped by unit", labelpad=3); ax.set_ylabel(f"{len(pu)} positive frames", labelpad=3)
ax.text(len(nu) / 2, len(pu) + 0.6, "29 of 144 pairs are within-unit ($\\alpha$ = 0.20)", ha="center", fontsize=6.5)
ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=C["W"]), Rectangle((0, 0), 1, 1, fc="#DDDDDD")], labels=["within-unit pair", "cross-unit pair"],
          loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=1.0, columnspacing=1.2)
letter(fig, 0.665, 0.445, "d")
ax = fig.add_axes([0.705, 0.02, 0.285, 0.42]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
spec = [("$E_1$  pooled", "all positive–negative pairs", C["E1"], None, None),
        ("$X$  cross-unit", "pairs from different units", "#555555", None, None),
        ("$W$  within-unit, pair-weighted", "unit weight $\\propto n_u^+ n_u^-$", C["W"], [15, 6, 8], 29),
        ("$E_2$  equal-weight mixed-unit", "equal weight per mixed-label unit", C["E2"], [1, 1, 1], 3)]
for r_, (t, d_, col, num, den) in enumerate(spec):
    yy = 9.2 - r_ * 2.05
    ax.text(0, yy, t, fontsize=6.8, va="center", color=col); ax.text(0, yy - 0.6, d_, fontsize=6, va="center", color="#555555")
    if num:
        xs = 0
        for k, nk in enumerate(num):
            wk = nk / den
            ax.add_patch(Rectangle((xs, yy - 1.55), wk * 9.6, 0.62, fc=col, alpha=[1, .72, .48][k], ec="white", lw=0.6))
            ax.text(xs + wk * 4.8, yy - 1.24, f"U{k+1}: {nk}/{den}", fontsize=6, ha="center", va="center", color="white" if k == 0 else "#111111")
            xs += wk * 9.6
ax.text(0, 0.7, "$E_1 = \\alpha\\, W + (1-\\alpha)\\, X$", fontsize=7.5)
save(fig, "Fig2_framework")
print("ok")
