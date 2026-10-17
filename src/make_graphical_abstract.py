"""Graphical abstract (MIA minimum 1328 x 531 px). python make_graphical_abstract.py <frozen_dir> <sages_results_dir> <out_dir> <endoscapes_paired_contrast_csv>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

ROOT, SRES, OUT = sys.argv[1:4]
FR = os.path.join(OUT, "frames")
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
B = pd.read_csv(f"{ROOT}/Tables/SAGES_10000_bootstrap.csv"); B["c"] = B.criterion.str.upper(); B["s"] = B.model.str.extract(r"seed(\d+)").astype(int)
A = pd.read_csv(f"{ROOT}/Tables/All_18_estimates.csv")
W_COL, POS, NEG = "#009E73", "#D55E00", "#56B4E9"
DS = {"Endoscapes": "#3D3D3D", "SAGES 2024": "white"}

fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
fig.text(0.015, 0.955, "Which frame pairs does a surgical-video AUROC compare?", fontsize=16, fontweight="bold", va="top")
# 1. frames
for k, (t, lab, col) in enumerate([("196_27400", "operation A, C1 negative", NEG), ("192_23825", "operation B, C1 positive", POS)]):
    a_ = fig.add_axes([0.015, 0.47 - k * 0.38, 0.17, 0.30]); a_.imshow(Image.open(f"{FR}/{t}.png")); a_.set_xticks([]); a_.set_yticks([])
    for sp in a_.spines.values(): sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(3)
    a_.set_title(lab, fontsize=11, pad=3)
fig.patches.append(FancyArrowPatch((0.10, 0.455), (0.10, 0.39), transform=fig.transFigure, arrowstyle="<|-|>", mutation_scale=12, color="#555555", lw=1.3))
fig.text(0.19, 0.42, "a cross-operation pair:\nused by the pooled AUROC,\nnot by the within-unit AUROC", fontsize=10.5, va="center")
# 2. pair matrix
ax = fig.add_axes([0.355, 0.24, 0.15, 0.52])
units = [(3, 5), (1, 6), (4, 2), (0, 5)]
pu = [u for u, (p, q) in enumerate(units) for _ in range(p)]; nu = [u for u, (p, q) in enumerate(units) for _ in range(q)]
for i, a in enumerate(pu):
    for j, b in enumerate(nu):
        ax.add_patch(Rectangle((j, len(pu) - 1 - i), 1, 1, fc=W_COL if a == b else "#DDDDDD", ec="white", lw=1))
ax.set_xlim(0, len(nu)); ax.set_ylim(0, len(pu)); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
ax.set_xlabel("negative frames, by unit"); ax.set_ylabel("positive frames")
ax.set_title("green: same unit\ngrey: different units", fontsize=11)
ax.text(len(nu) / 2, -4.2, "$E_1=\\alpha W+(1-\\alpha)X$", fontsize=14, ha="center")
# 3. alpha
ax2 = fig.add_axes([0.575, 0.24, 0.13, 0.52])
al = A.groupby(["dataset", "criterion"]).alpha.first() * 100
x = np.arange(3)
for d, off in (("Endoscapes", -0.2), ("SAGES 2024", 0.2)):
    v = [al[(d, c)] for c in ["C1", "C2", "C3"]]
    ax2.bar(x + off, v, 0.4, color=DS[d], hatch=None if d == "Endoscapes" else "////", edgecolor="#3D3D3D", lw=0.8, label=d.replace(" 2024", ""))
ax2.set_xticks(x); ax2.set_xticklabels(["C1", "C2", "C3"]); ax2.set_ylabel("within-unit pairs, $\\alpha$ (%)")
ax2.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, handlelength=1.0, columnspacing=0.8)
ax2.set_ylim(0, 3.5)
# 4. primary contrasts with unit-level 95% CIs
EC = pd.read_csv(sys.argv[4]); EC["criterion"] = EC.criterion.str.upper()
ax3 = fig.add_axes([0.80, 0.16, 0.19, 0.64])
y = 0; yt = []; yl = []
for c in ["C1", "C2", "C3"]:
    for s in [17, 29, 43]:
        e = EC[(EC.criterion == c) & (EC.seed == s) & (EC.contrast == "W-E1")].iloc[0]
        ax3.errorbar(e.point, y + 0.15, xerr=[[e.point - e.ci_low], [e.ci_high - e.point]], fmt="D", color="#3D3D3D", ms=4.5, lw=1.2)
        r = B[(B.c == c) & (B.s == s)].iloc[0]
        ax3.errorbar(r.W_minus_E1, y - 0.15, xerr=[[r.W_minus_E1 - r.W_minus_E1_lo], [r.W_minus_E1_hi - r.W_minus_E1]], fmt="o", color=W_COL, ms=5, lw=1.2)
        yt.append(y); yl.append(f"{c} seed {s}"); y -= 1
    y -= 0.5
ax3.axvline(0, color="#444444", lw=0.8); ax3.set_yticks(yt); ax3.set_yticklabels(yl, fontsize=9.5)
ax3.set_xlabel("$W-E_1$ with 95% CI")
from matplotlib.lines import Line2D
ax3.legend(handles=[Line2D([], [], marker="D", ls="-", color="#3D3D3D", ms=4.5), Line2D([], [], marker="o", ls="-", color=W_COL, ms=5)],
           labels=["Endoscapes\n(original)", "SAGES\n(retrained)"], loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False, fontsize=10,
           handletextpad=0.3, columnspacing=1.0)
fig.text(0.985, 0.955, "Endoscapes images: CC BY-NC-SA 4.0 (Mascagni et al., 2025)", fontsize=8, ha="right", va="top", color="#555555")
fig.text(0.015, 0.035, "Fewer than 3% of compared pairs came from the same unit and $W-E_1$ changed sign: report $\\alpha$, "
         "a within-unit AUROC with unit-level intervals and a simple within-unit baseline.", fontsize=11.5, color="#222222")
fig.savefig(f"{OUT}/Graphical_abstract.png", dpi=200); fig.savefig(f"{OUT}/Graphical_abstract.pdf")
