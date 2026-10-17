"""Fig. 5 (pair composition) and Fig. 8 (reference-standard sensitivity).
python make_figures.py <frozen_dir> <out_dir> <endoscapes_annotator_sensitivity.csv>"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT, OUT = sys.argv[1], sys.argv[2]
MM = 1 / 25.4
FULL = 190 * MM
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
DS = {"Endoscapes": "#3D3D3D", "SAGES 2024": "white"}
HATCH = {"Endoscapes": None, "SAGES 2024": "/////"}
plt.rcParams["hatch.linewidth"] = 0.5
SEED_MK = {17: "o", 29: "s", 43: "^"}


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("pdf", "png", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02,
                    dpi=600 if ext != "png" else 300, **kw)
    plt.close(fig)


A = pd.read_csv(f"{ROOT}/Tables/All_18_estimates.csv")
B = pd.read_csv(f"{ROOT}/Tables/SAGES_10000_bootstrap.csv")
B["criterion"] = B.criterion.str.upper(); B["seed"] = B.model.str.extract(r"seed(\d+)").astype(int)
R = pd.read_csv(f"{ROOT}/Tables/Rater_sensitivity.csv")
R["criterion"] = R.criterion.str.upper(); R["seed"] = R.model.str.extract(r"seed(\d+)").astype(int)
P = pd.read_csv(f"{ROOT}/Tables/Pair_disagreement.csv"); P["criterion"] = P.criterion.str.upper()
crit = ["C1", "C2", "C3"]; x = np.arange(3)

# Fig. 5  pair composition, eligibility, weight concentration
fig, axs = plt.subplots(1, 3, figsize=(FULL, 58 * MM), gridspec_kw=dict(wspace=0.42))
al = A.groupby(["dataset", "criterion"]).alpha.first()
el = A.groupby(["dataset", "criterion"]).eligible.first()
ax = axs[0]; panel(ax, "a", x=-0.28)
for d, off in (("Endoscapes", -0.12), ("SAGES 2024", 0.12)):
    v = [al[(d, c)] * 100 for c in crit]
    ax.plot(x + off, v, "D" if d == "Endoscapes" else "o", color="#3D3D3D" if d == "Endoscapes" else "#8C8C8C", ms=4.5, label=d)
    for xi, vi in zip(x + off, v):
        ax.text(xi + 0.07, vi, f"{vi:.2f}", va="center", fontsize=6)
ax.set_yscale("log"); ax.set_ylim(0.05, 10); ax.set_xlim(-0.5, 2.6); ax.set_xticks(x); ax.set_xticklabels(crit)
ax.set_ylabel("Within-unit pairs, $\\alpha$ (%)")
ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"])
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.04), ncol=2, columnspacing=0.8, handletextpad=0.2)
ax = axs[1]; panel(ax, "b", x=-0.28)
den = {"Endoscapes": 40, "SAGES 2024": 293}
for d, off in (("Endoscapes", -0.18), ("SAGES 2024", 0.18)):
    v = [el[(d, c)] / den[d] * 100 for c in crit]
    ax.bar(x + off, v, 0.36, color=DS[d], hatch=HATCH[d], edgecolor="#3D3D3D", lw=0.6)
    for xi, c, vi in zip(x + off, crit, v):
        ax.text(xi, vi + 1.5, f"{int(el[(d, c)])}/{den[d]}", ha="center", fontsize=6, rotation=90, va="bottom")
ax.set_ylim(0, 100); ax.set_xticks(x); ax.set_xticklabels(crit)
ax.set_ylabel("Units with both labels (%)")
ax = axs[2]; panel(ax, "c", x=-0.28)
for c, ls in zip(crit, ["-", "--", ":"]):
    w = np.sort(P[P.criterion == c].within_pairs.values)[::-1]
    cum = np.concatenate([[0], np.cumsum(w) / w.sum()])
    neff = w.sum() ** 2 / (w ** 2).sum()
    ax.plot(np.arange(len(cum)), cum, ls, color=DS["Endoscapes"], lw=1.1,
            label=f"{c}: {len(w)} operations, $n_{{\\mathrm{{eff}}}}$ = {neff:.1f}")
ax.set_xlabel("Mixed-label operations, ranked by weight")
ax.set_ylabel("Cumulative share of $W$ weight")
ax.set_ylim(0, 1.02); ax.set_xlim(0, 24.5)
ax.legend(loc="lower right", title="Endoscapes", title_fontsize=6.5)
save(fig, "Fig5_pair_composition")

# Fig. 8  reference-standard sensitivity: Endoscapes annotators (top), SAGES rating columns (bottom)
EA = pd.read_csv(sys.argv[3])
fig, axs = plt.subplots(2, 3, figsize=(FULL, 98 * MM), sharey="row", gridspec_kw=dict(wspace=0.08, hspace=0.55))
jit = {17: -0.1, 29: 0.0, 43: 0.1}
MFC = {17: "#009E73", 29: "white", 43: "#7FCEB9"}
for r_, ds in enumerate(("Endoscapes", "SAGES 2024")):
    for ax, c, L in zip(axs[r_], crit, ("abc", "def")[r_]):
        panel(ax, L, x=-0.04 if L not in "ad" else -0.27)
        for s in (17, 29, 43):
            if ds == "Endoscapes":
                g = EA[(EA.criterion == c) & (EA.seed == s)].set_index("reference")
                v = [g.loc[f"annotator {k}", "W_minus_E1"] for k in (1, 2, 3)] + [g.loc["majority", "W_minus_E1"]]
            else:
                v = [R[(R.criterion == c) & (R.seed == s) & (R.rater == k)].W_minus_E1.iloc[0] for k in (1, 2, 3)]
                v.append(B[(B.criterion == c) & (B.seed == s)].W_minus_E1.iloc[0])
            ax.plot(np.arange(4) + jit[s], v, ls="none", marker=SEED_MK[s], ms=3.8, color="#009E73", mfc=MFC[s], mew=0.8, label=f"seed {s}")
        ax.axhline(0, color="#666666", lw=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["ann. 1", "ann. 2", "ann. 3", "majority"] if ds == "Endoscapes" else ["col. 1", "col. 2", "col. 3", "majority"])
        ax.set_xlim(-0.35, 3.35); ax.set_title(f"{ds.replace(' 2024', '')}, {c}", fontsize=7.5)
        ax.yaxis.grid(True, lw=0.4, color="#E8E8E8"); ax.set_axisbelow(True)
    axs[r_, 0].set_ylabel("$W-E_1$ (AUROC)")
axs[0, 1].set_xlabel("Reference standard (single annotator or majority vote)")
axs[1, 1].set_xlabel("Reference standard (single rating column or majority vote)")
axs[0, 0].legend(loc="lower left", ncol=3, columnspacing=0.6, handletextpad=0.2, bbox_to_anchor=(0, 1.12))
save(fig, "Fig8_rater_sensitivity")
print("figures written")
