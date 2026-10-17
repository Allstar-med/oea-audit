"""Fig. 1: from the clinical question to the scientific endpoint, on real data.
python make_fig1.py <data_dir> <reference_anchored.csv> <out_dir>
<data_dir> is the repository data directory (endoscapes/, sages/, frozen/, figure_frames/)."""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.lines import Line2D
from PIL import Image

D, REF, OUT = sys.argv[1:4]
MM = 1 / 25.4
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False, "hatch.linewidth": 0.5,
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "mathtext.bf": "Liberation Sans:bold"})
POS, NEG, WCOL, E2COL, INK = "#D55E00", "#56B4E9", "#009E73", "#CC79A7", "#222222"
ENDO, SAG = "#3D3D3D", "#3D3D3D"
L = pd.read_csv(f"{D}/endoscapes/endoscapes_test_linkage.csv")
S = pd.read_csv(f"{D}/sages/sages_labels.csv"); S = S[S.primary_include == 1]
A = pd.read_csv(f"{D}/frozen/Tables/All_18_estimates.csv")
B = pd.read_csv(f"{D}/frozen/Tables/SAGES_10000_bootstrap.csv"); B["c"] = B.criterion.str.upper(); B["s"] = B.model.str.extract(r"seed(\d+)").astype(int)
ST = pd.read_csv(f"{D}/sages/results/source_test_estimands.csv")
EC = pd.read_csv(f"{D}/endoscapes/Endoscapes_10000_paired_contrast_CI.csv"); EC["c"] = EC.criterion.str.upper()
R = pd.read_csv(REF)
al = A.groupby(["dataset", "criterion"]).alpha.first() * 100
el = A.groupby(["dataset", "criterion"]).eligible.first()
FE = f"{D}/figure_frames/endoscapes"; FS = f"{D}/figure_frames/sages"


def img_endo(v, f, w=320):
    im = Image.open(f"{FE}/{v}_{f}.jpg").convert("RGB")
    return im.resize((w, int(w * im.size[1] / im.size[0])))


def img_sages(name, w=320):
    im = Image.open(f"{FS}/{name}").convert("RGB")
    a = np.asarray(im).astype(int).sum(2)
    r = np.where((a > 60).mean(1) > 0.2)[0]; c = np.where((a > 60).mean(0) > 0.2)[0]
    im = im.crop((c.min(), r.min(), c.max() + 1, r.max() + 1)); x, y = im.size
    m = int(0.015 * x); im = im.crop((m, m, x - m, y - m)); x, y = im.size
    tw = int(y * 854 / 480)
    if tw < x:
        im = im.crop(((x - tw) // 2, 0, (x - tw) // 2 + tw, y))
    return im.resize((w, int(w * im.size[1] / im.size[0])))


def framed(ax, im, col, lw=1.4):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(col); s.set_linewidth(lw)


def letter(x, y, L, title):
    fig.text(x, y, f"({L})", fontsize=8.5, fontweight="bold", va="top")
    fig.text(x + 0.028, y - 0.001, title, fontsize=7.5, fontweight="bold", va="top", color=INK)


def box(x, y, w, h, fc="#F6F7F9"):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.008", transform=fig.transFigure,
                                      fc=fc, ec="#D5DAE1", lw=0.6, zorder=-10))


fig = plt.figure(figsize=(190 * MM, 228 * MM))

# ---------------------------------------------------------------- (a) clinical question
box(0.0, 0.775, 0.60, 0.222)
letter(0.008, 0.993, "a", "Clinical question versus benchmark practice")
g = L[L.video == 196].sort_values("frame").reset_index(drop=True)
t = (g.frame - g.frame.min()) / 25.0
picks = [27400, 28650, 29900, 32525, 35025]
fig.text(0.015, 0.962, "Within one operation (Endoscapes op. 196):\nis C1 achieved at this moment?", fontsize=6.5, color=INK, va="top")
for k, f in enumerate(picks):
    ax = fig.add_axes([0.015 + k * 0.074, 0.868, 0.068, 0.058])
    y = int(g.loc[g.frame == f, "y_c1"].iloc[0])
    framed(ax, img_endo(196, f), POS if y else NEG)
    ax.set_title(f"{(f - g.frame.min()) / 25:.0f} s", fontsize=6.3, pad=1.5)
    if f in (27400, 35025):
        ax.text(0.04, 0.93, " N1 " if f == 27400 else " N2 ", transform=ax.transAxes, fontsize=6, color="white", fontweight="bold", va="top",
                bbox=dict(boxstyle="square,pad=0.1", fc=NEG, ec="none"))
axl = fig.add_axes([0.015, 0.846, 0.362, 0.012])
for i in range(len(g)):
    axl.add_patch(Rectangle((t[i] - 2.5, 0), 5, 1, fc=POS if g.y_c1[i] else NEG, ec="none"))
axl.set_xlim(t.min() - 2.5, t.max() + 2.5); axl.set_ylim(0, 1); axl.axis("off")
fig.text(0.015, 0.842, "C1 label at the 75 annotated frames; C1 score (0–1) of seed 29", fontsize=6.3, color="#444444", va="top")
axs = fig.add_axes([0.015, 0.79, 0.362, 0.036])
axs.plot(t, g.p_c1_seed29, color="#666666", ls="--", lw=0.8, marker="s", ms=1.2); axs.set_xlim(t.min() - 2.5, t.max() + 2.5); axs.set_ylim(-0.05, 1.05)
axs.set_yticks([0, 1]); axs.set_xticks([0, 100, 200, 300, 370]); axs.set_xticklabels(["0", "100", "200", "300", "370 s"]); axs.tick_params(labelsize=6, pad=1)
axs.set_yticks([]); axs.spines["left"].set_visible(False)
fig.text(0.392, 0.962, "Benchmark practice: frames pooled\nfrom all 40 test operations", fontsize=6.5, color=INK, va="top")
mosaic = [("e", 185, 12750), ("e", 180, 19600), ("e", 177, 40825), ("e", 199, 27375), ("e", 169, 11675), ("e", 174, 23950),
          ("e", 188, 22500), ("e", 199, 27875), ("e", 192, 24575)]
lab_s = pd.read_csv(f"{FS}/index.csv") if os.path.exists(f"{FS}/index.csv") else None
for k, m in enumerate(mosaic):
    r_, c_ = divmod(k, 3)
    ax = fig.add_axes([0.392 + c_ * 0.068, 0.902 - r_ * 0.040, 0.063, 0.036])
    if m[0] == "e":
        y = int(L[(L.video == m[1]) & (L.frame == m[2])].y_c1.iloc[0]); framed(ax, img_endo(m[1], m[2], 200), POS if y else NEG, 1.0)
    else:
        row = lab_s[lab_s.file == m[1]].iloc[0]; y = int(row.c1_r1 + row.c1_r2 + row.c1_r3 >= 2)
        framed(ax, img_sages(m[1], 200), POS if y else NEG, 1.0)
fig.text(0.392, 0.781, "one AUROC over all pairs,\nmostly from different operations", fontsize=6, color="#444444", va="bottom")
fig.patches.append(FancyArrowPatch((0.378, 0.895), (0.389, 0.895), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, color="#777777"))
fig.legend(handles=[Rectangle((0, 0), 1, 1, fc="white", ec=POS, lw=1.4), Rectangle((0, 0), 1, 1, fc="white", ec=NEG, lw=1.4)],
           labels=["C1 positive", "C1 negative"], loc="upper right", bbox_to_anchor=(0.598, 0.997), ncol=2, fontsize=6, handlelength=1.0,
           handletextpad=0.4, columnspacing=0.8)

# ---------------------------------------------------------------- (b) data structure
box(0.61, 0.775, 0.39, 0.222)
letter(0.618, 0.993, "b", "Data structure: frames nested in units")
endo_units = [gg.sort_values("frame").y_c1.values for _, gg in L.groupby("video")]
endo_units.sort(key=len)
ax = fig.add_axes([0.625, 0.805, 0.17, 0.105])
for r_, u in enumerate(endo_units):
    for c_, v in enumerate(u):
        ax.add_patch(Rectangle((c_, r_), 1, 1, fc=POS if v else NEG, ec="none"))
ax.set_xlim(0, max(len(u) for u in endo_units)); ax.set_ylim(0, len(endo_units)); ax.set_xticks([0, 50, 100, 150]); ax.set_yticks([])
ax.tick_params(labelsize=6); ax.set_xlabel("annotated frame (5 s apart)", fontsize=6, labelpad=1); ax.set_ylabel("40 operations", fontsize=6, labelpad=1)
ax.set_title(f"Endoscapes\n40 operations, 1,663 frames\nC1 mixed-label {int(el[('Endoscapes', 'C1')])}/40; $\\alpha$ = {al[('Endoscapes', 'C1')]:.2f}%", fontsize=6, pad=2)
sg_units = [gg.sort_values("frame_id").c1_majority.values for _, gg in S.groupby("clip_id")]
sg_units.sort(key=lambda v: (v.sum() > 0, v.mean()))
ax = fig.add_axes([0.82, 0.805, 0.165, 0.105])
arr = np.array(sg_units)
ax.imshow(np.where(arr[:, :, None] == 1, np.array([213, 94, 0]) / 255, np.array([86, 180, 233]) / 255), aspect="auto", interpolation="nearest",
          origin="lower", extent=[0, 18, 0, len(sg_units)])
ax.set_xticks([0, 9, 18]); ax.set_yticks([]); ax.tick_params(labelsize=6); ax.set_xlabel("annotated frame (5 s apart)", fontsize=6, labelpad=1)
ax.set_ylabel("293 clips", fontsize=6, labelpad=1)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.set_title(f"SAGES 2024\n293 clips, 5,274 frames\nC1 mixed-label {int(el[('SAGES 2024', 'C1')])}/293; $\\alpha$ = {al[('SAGES 2024', 'C1')]:.2f}%", fontsize=6, pad=2)
fig.text(0.625, 0.966, "each row is one unit (operation or clip)", fontsize=6, color="#444444", va="top")

# ---------------------------------------------------------------- (c) method
box(0.0, 0.535, 0.40, 0.232)
letter(0.008, 0.763, "c", "Method: estimands and audit")
ax = fig.add_axes([0.025, 0.60, 0.125, 0.12])
units = [(3, 5), (1, 6), (4, 2), (0, 5)]
pu = [u for u, (p, q) in enumerate(units) for _ in range(p)]; nu = [u for u, (p, q) in enumerate(units) for _ in range(q)]
for i, a in enumerate(pu):
    for j, b in enumerate(nu):
        ax.add_patch(Rectangle((j, len(pu) - 1 - i), 1, 1, fc=WCOL if a == b else "#DDDDDD", ec="white", lw=0.6))
ax.set_xlim(0, len(nu)); ax.set_ylim(0, len(pu)); ax.set_aspect("equal"); ax.axis("off")
ax.text(len(nu) / 2, -1.2, "negative frames by unit", ha="center", va="top", fontsize=6)
ax.text(-0.8, len(pu) / 2, "positive frames", rotation=90, ha="right", va="center", fontsize=6)
ax.text(len(nu) / 2, len(pu) + 0.6, "toy test set, 4 units", ha="center", fontsize=6)
tx = 0.175
lines = [("$E_1$  pooled AUROC, all pairs", INK), ("$X$  cross-unit pairs only", "#777777"),
         ("$W$  within-unit pairs, weight $n_u^+n_u^-$", WCOL), ("$E_2$  mean over mixed-label units", E2COL)]
for k, (s, c) in enumerate(lines):
    fig.text(tx, 0.725 - k * 0.022, s, fontsize=6.5, color=c, va="center")
fig.text(tx, 0.632, "$E_1=\\alpha W+(1-\\alpha)X$", fontsize=8.5, va="center")
fig.text(tx, 0.606, "$\\Delta W=W(\\mathrm{model})-W(\\mathrm{time\\ baseline})$", fontsize=7.2, va="center")
fig.text(tx, 0.582, "unit-level bootstrap, matched across models;\n$t$ interval with $n_{\\mathrm{eff}}-1$ df for $W$", fontsize=6, va="center", color="#444444")
fig.text(0.02, 0.548, "green: within-unit pairs; grey: cross-unit pairs. Only $W$ and $E_2$ are\ninvariant to unit-specific score shifts (Proposition 2).", fontsize=6, color="#444444")

# ---------------------------------------------------------------- (d) real case
box(0.41, 0.535, 0.59, 0.232)
letter(0.418, 0.763, "d", "Real case: four Endoscapes test frames for C1")
PR = {"N1": (196, 27400), "N2": (196, 35025), "P1": (192, 23825), "P2": (192, 24575)}
pos_ax = {}
for k, (tag, (v, f)) in enumerate(PR.items()):
    x0 = 0.425 + k * 0.143 + (0.012 if k >= 2 else 0)
    ax = fig.add_axes([x0, 0.632, 0.128, 0.098])
    r = L[(L.video == v) & (L.frame == f)].iloc[0]
    framed(ax, img_endo(v, f, 360), POS if r.y_c1 else NEG)
    ax.text(0.03, 0.95, f" {tag} ", transform=ax.transAxes, fontsize=6.5, color="white", fontweight="bold", va="top",
            bbox=dict(boxstyle="square,pad=0.15", fc=POS if r.y_c1 else NEG, ec="none"))
    ax.text(0.5, -0.05, f"votes {int(round(r.official_c1 * 3))}/3\nseed 17: {r.p_c1_seed17:.2f}\nseed 29: {r.p_c1_seed29:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=6.2)
    pos_ax[tag] = ax
fig.text(0.425 + 0.135, 0.742, "operation 196: mixed-label (40 negative, 35 positive frames)", ha="center", fontsize=6, color="#444444")
fig.text(0.725 + 0.135, 0.742, "operation 192: positive frames only", ha="center", fontsize=6, color="#444444")
fig.text(0.425, 0.588, "The four positive–negative pairs (P1–N1, P1–N2, P2–N1, P2–N2) are cross-operation pairs:\nthey count in $E_1$ and $X$ but not in $W$.", fontsize=6, color=INK, va="top")
fig.text(0.425, 0.560, "Seed 17 ordered all four incorrectly and seed 29 all four correctly, although within\noperation 196 the two models ranked frames almost equally well (AUROC 0.80 and 0.81).", fontsize=6, color=INK, va="top")

# ---------------------------------------------------------------- (e) finding: Endoscapes
y0, h0 = 0.128, 0.335
box(0.0, 0.07, 0.325, 0.457)
letter(0.008, 0.522, "e", "Pooled and within-unit AUROC differ")
ax = fig.add_axes([0.095, y0, 0.215, h0])
crit = ["C1", "C2", "C3"]; seeds = [17, 29, 43]
yy = 0; yt, yl = [], []
for c in crit:
    for s in seeds:
        r = A[(A.dataset == "Endoscapes") & (A.criterion == c) & (A.seed == s)].iloc[0]
        e = EC[(EC.c == c) & (EC.seed == s) & (EC.contrast == "W-E1")].iloc[0]
        ax.plot([r.E1, r.W], [yy, yy], color="#AAAAAA", lw=1.0, zorder=1)
        ax.plot(r.E1, yy, "o", color=INK, ms=3.4, zorder=3); ax.plot(r.W, yy, "o", color=WCOL, ms=3.4, zorder=3)
        yt.append(yy); yl.append(f"{c}  seed {s}"); yy -= 1
    yy -= 0.6
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.3); ax.set_xlabel("AUROC"); ax.set_xlim(0.57, 0.80)
ax.xaxis.grid(True, lw=0.4, color="#E5E5E5"); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, ms=3.4), Line2D([], [], marker="o", ls="", color=WCOL, ms=3.4)],
          labels=["$E_1$ (pooled)", "$W$ (within-unit)"], loc="lower center", bbox_to_anchor=(0.45, 1.005), ncol=2, fontsize=6, handletextpad=0.3)
fig.text(0.015, 0.497, "Endoscapes, original checkpoints", fontsize=6.3, color="#444444", va="top")
fig.text(0.015, 0.078, "$\\alpha$ = 2.75–2.97%; $W-E_1$ from −0.074 to +0.126", fontsize=6, color="#444444")

# ---------------------------------------------------------------- (f) transfer: SAGES
box(0.335, 0.07, 0.325, 0.457)
letter(0.343, 0.522, "f", "Transfer to a second dataset")
fig.text(0.35, 0.497, "Retrained checkpoints: SAGES 2024 and Endoscapes", fontsize=6.3, color="#444444", va="top")
ax = fig.add_axes([0.43, y0, 0.215, h0])
yy = 0; yt, yl = [], []
for c in crit:
    for s in seeds:
        r = B[(B.c == c) & (B.s == s)].iloc[0]
        exc = r.W_minus_E1_lo > 0 or r.W_minus_E1_hi < 0
        ax.errorbar(r.W_minus_E1, yy - 0.15, xerr=[[r.W_minus_E1 - r.W_minus_E1_lo], [r.W_minus_E1_hi - r.W_minus_E1]], fmt="o", ms=3.2,
                    color=WCOL, mfc=WCOL if exc else "white", elinewidth=0.9)
        e = ST[(ST.criterion == c) & (ST.seed == s)].iloc[0]
        ax.plot(e.W - e.E1, yy + 0.15, "D", color=INK, ms=2.8)
        yt.append(yy); yl.append(f"{c}  seed {s}"); yy -= 1
    yy -= 0.6
ax.axvline(0, color="#666666", lw=0.6); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.3)
ax.set_xlabel("$W-E_1$"); ax.xaxis.grid(True, lw=0.4, color="#E5E5E5"); ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace("-", "−") if abs(v) > 1e-9 else "0"))
ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=WCOL, ms=3.2, lw=0.9), Line2D([], [], marker="D", ls="", color=INK, ms=2.8)],
          labels=["SAGES, 95% CI", "Endoscapes"], loc="lower center", bbox_to_anchor=(0.45, 1.005), ncol=2, fontsize=6, handletextpad=0.3)
fig.text(0.35, 0.078, "$\\alpha$ = 0.17–0.19%; open: CI includes 0; sign agreed in 7 of 9", fontsize=6, color="#444444")

# ---------------------------------------------------------------- (g) endpoint
box(0.67, 0.07, 0.33, 0.457)
letter(0.678, 0.522, "g", "Within-unit AUROC versus elapsed time")
fig.text(0.685, 0.497, "$\\Delta W$ = $W$(model) − $W$(time within recording),\n95% $t$ interval from matched unit resamples", fontsize=6.3, color="#444444", va="top")
ax = fig.add_axes([0.765, y0, 0.22, h0])
yy = 0; yt, yl = [], []
for c in crit:
    for ds, off in (("Endoscapes", 0.17), ("SAGES 2024", -0.17)):
        for s in seeds:
            r = R[(R.dataset == ds) & (R.criterion == c) & (R.seed == s)].iloc[0]
            exc = r.dW_tlo > 0 or r.dW_thi < 0
            ax.errorbar(r.dW, yy, xerr=[[r.dW - r.dW_tlo], [r.dW_thi - r.dW]], fmt="D" if ds == "Endoscapes" else "o", ms=3.0,
                        color=ENDO if ds == "Endoscapes" else "#7A7A7A", mfc=(ENDO if ds == "Endoscapes" else "#7A7A7A") if exc else "white",
                        elinewidth=0.8)
            yt.append(yy); yl.append(f"{c} {'Endo.' if ds == 'Endoscapes' else 'SAGES'} {s}"); yy -= 0.62
        yy -= 0.25
    yy -= 0.35
ax.axvline(0, color="#666666", lw=0.6); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5.6)
ax.set_xlabel("$\\Delta W$"); ax.xaxis.grid(True, lw=0.4, color="#E5E5E5"); ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "−") if abs(v) > 1e-9 else "0"))
ax.text(0.02, 1.01, "\u2190 worse than time", transform=ax.transAxes, fontsize=6, color="#444444", va="bottom")
ax.text(0.98, 1.01, "better \u2192", transform=ax.transAxes, fontsize=6, color="#444444", va="bottom", ha="right")
fig.text(0.685, 0.078, "diamonds Endoscapes, circles SAGES; open: includes 0", fontsize=6, color="#444444")

# ---------------------------------------------------------------- (h) evidence chain
box(0.0, 0.0, 1.0, 0.062, fc="#FFFFFF")
letter(0.008, 0.058, "h", "Evidence chain")
steps = ["SHA-256 match for 1,663/1,663\nimages and 1,000/1,000 videos",
         "duplicate rule fixed before training;\nmodel manifest before inference",
         "checkpoint hashes verified at\ninference; no SAGES in training",
         "two independent implementations\n(point estimates within $5\\times10^{-16}$)",
         "public code reproduces every\nrow-level result exactly"]
for k, s in enumerate(steps):
    x = 0.012 + k * 0.198
    fig.patches.append(FancyBboxPatch((x, 0.006), 0.182, 0.032, boxstyle="round,pad=0.0,rounding_size=0.006", transform=fig.transFigure,
                                      fc="#EEF6F2", ec="#9CCFB8", lw=0.6))
    fig.text(x + 0.091, 0.022, s, ha="center", va="center", fontsize=5.9, color=INK)

for ext in ("pdf", "png", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(os.path.join(OUT, f"Fig1_overview.{ext}"), bbox_inches="tight", pad_inches=0.03, dpi=300 if ext == "png" else 600, **kw)
print("Fig1 written")
