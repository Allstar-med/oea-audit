"""Figures built from original images.
  - copies the four Fig. 1 frames from the verified Endoscapes JPEGs (SHA-256 checked) into figures/frames
  - Fig. 3: dataset overview (Endoscapes + SAGES, stratified by number of criteria achieved)
  - Fig. 9: within-operation trajectories (operation 196) + per-operation ranking disagreement
python make_image_figs.py <endoscapes_test_dir> <sages_examples_dir> <linkage_csv> <sages_labels_csv> <pair_table_csv> <out_fig_dir>"""
import sys, os, hashlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from PIL import Image

ED, SD, LINK, SLAB, PAIRS, OUT = sys.argv[1:7]
MM = 1 / 25.4
plt.rcParams["hatch.linewidth"] = 0.5
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.3, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.frameon": False, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
POS, NEG = "#D55E00", "#56B4E9"
ENDO_C, SAGES_C = "#3D3D3D", "white"
SEEDS = {17: dict(color="#000000", ls="-", marker="o"), 29: dict(color="#666666", ls="--", marker="s"), 43: dict(color="#AAAAAA", ls="-.", marker="^")}
L = pd.read_csv(LINK)


def save(fig, name):
    for ext in ("pdf", "png", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=300 if ext == "png" else 600, **kw)
    plt.close(fig)


def endo_img(v, f, check=True):
    p = os.path.join(ED, f"{v}_{f}.jpg")
    if check:
        r = L[(L.video == v) & (L.frame == f)].iloc[0]
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == r.image_sha256, (v, f)
    return Image.open(p).convert("RGB")


# ---- 1. original frames for Fig. 1 / Fig. 2 / graphical abstract
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
for v, f in [(196, 27400), (196, 35025), (192, 23825), (192, 24575)]:
    endo_img(v, f).save(os.path.join(OUT, "frames", f"{v}_{f}.png"))

# ---- 2. Fig. 3 dataset overview
L["k"] = L.y_c1 + L.y_c2 + L.y_c3
rng = np.random.default_rng(20261017)
endo_pick = {}
for k in range(4):
    g = L[L.k == k]
    ops = rng.permutation(np.sort(g.video.unique()))[:2]
    endo_pick[k] = [g[g.video == o].sample(1, random_state=int(rng.integers(1e9))).iloc[0] for o in ops]
S = pd.read_csv(os.path.join(SD, "index.csv"))
sages_pick = {k: [S[S.file.str.startswith(f"k{k}_0{j}")].iloc[0] for j in (1, 2)] for k in range(4)}
SL = pd.read_csv(SLAB); SL = SL[SL.primary_include == 1]
fig = plt.figure(figsize=(190 * MM, 104 * MM))
gs = fig.add_gridspec(4, 4, left=0.085, right=0.70, top=0.93, bottom=0.02, wspace=0.04, hspace=0.24)


def tag(ax, votes):
    for i, (name, v) in enumerate(zip(("C1", "C2", "C3"), votes)):
        col = POS if v >= 2 else NEG
        ax.text(0.015 + i * 0.33, 0.035, f"{name} {v}/3", transform=ax.transAxes, fontsize=6, color="white", fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", fc=col, ec="none"), va="bottom")


for k in range(4):
    for j in range(2):
        r = endo_pick[k][j]
        ax = fig.add_subplot(gs[j, k]); ax.imshow(endo_img(int(r.video), int(r.frame))); ax.set_xticks([]); ax.set_yticks([])
        tag(ax, [int(round(r.official_c1 * 3)), int(round(r.official_c2 * 3)), int(round(r.official_c3 * 3))])
        ax.set_title(f"op. {int(r.video)}, frame {int(r.frame)}", fontsize=6, pad=1.5)
        if j == 0: ax.text(0.5, 1.30, f"{k} criteri{'on' if k == 1 else 'a'} achieved", transform=ax.transAxes, ha="center", fontsize=7, fontweight="bold")
        r = sages_pick[k][j]
        ax = fig.add_subplot(gs[2 + j, k])
        im = Image.open(os.path.join(SD, r.file)).convert("RGB")
        arr = np.asarray(im).astype(int).sum(axis=2); rows_ = np.where((arr > 60).mean(axis=1) > 0.2)[0]; cols_ = np.where((arr > 60).mean(axis=0) > 0.2)[0]
        im = im.crop((cols_.min(), rows_.min(), cols_.max() + 1, rows_.max() + 1)); w, h = im.size
        m = int(round(0.015 * w)); im = im.crop((m, m, w - m, h - m)); w, h = im.size
        tw = int(h * 854 / 480)
        if tw < w: im = im.crop(((w - tw) // 2, 0, (w - tw) // 2 + tw, h))
        elif tw > w:
            th_ = int(w * 480 / 854); im = im.crop((0, (h - th_) // 2, w, (h - th_) // 2 + th_))
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        tag(ax, [int(r.c1_r1 + r.c1_r2 + r.c1_r3), int(r.c2_r1 + r.c2_r2 + r.c2_r3), int(r.c3_r1 + r.c3_r2 + r.c3_r3)])
        ax.set_title(f"clip {r.clip_id[:8]}, {r.timestamp_s:.0f} s", fontsize=6, pad=1.5)
fig.text(0.035, 0.735, "Endoscapes\n(40 test operations)", rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
fig.text(0.035, 0.265, "SAGES 2024\n(293 test clips)", rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
fig.text(0.005, 0.99, "(a)", fontsize=8.5, fontweight="bold", va="top")
# (b) frames per unit
ax = fig.add_axes([0.775, 0.60, 0.215, 0.33])
fpu = L.groupby("video").size().values
ax.hist(fpu, bins=np.arange(0, 170, 10), color=ENDO_C)
assert len(fpu) == 40 and fpu.max() < 170
ax.axvline(18, color="#000000", lw=0.9, ls="--")
ax.text(21, ax.get_ylim()[1] * 0.95, "SAGES: 18 per clip", color="#000000", fontsize=6, va="top")
ax.set_xlim(0, 170)
ax.set_xlabel("Annotated frames per unit"); ax.set_ylabel("Number of operations")
ax.set_title("Endoscapes operations", fontsize=6.5, pad=3)
fig.text(0.715, 0.99, "(b)", fontsize=8.5, fontweight="bold", va="top")
# (c) prevalence
ax = fig.add_axes([0.775, 0.10, 0.215, 0.33])
x = np.arange(3)
ep = [L[f"y_c{i}"].mean() * 100 for i in (1, 2, 3)]; sp = [SL[f"c{i}_majority"].mean() * 100 for i in (1, 2, 3)]
ax.bar(x - 0.19, ep, 0.36, color=ENDO_C, label="Endoscapes"); ax.bar(x + 0.19, sp, 0.36, color=SAGES_C, hatch="/////", edgecolor="#3D3D3D", lw=0.6, label="SAGES 2024")
for xi, a, b in zip(x, ep, sp):
    ax.text(xi - 0.19, a + 0.8, f"{a:.1f}", ha="center", fontsize=6); ax.text(xi + 0.19, b + 0.8, f"{b:.1f}", ha="center", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(["C1", "C2", "C3"]); ax.set_ylabel("Positive frames (%)"); ax.set_ylim(0, 36)
ax.legend(loc="upper left", fontsize=6)
fig.text(0.715, 0.48, "(c)", fontsize=8.5, fontweight="bold", va="top")
save(fig, "Fig3_datasets")
sel = [dict(dataset="Endoscapes", stratum=k, unit=int(r.video), frame=int(r.frame), file=f"{int(r.video)}_{int(r.frame)}.jpg", sha256=r.image_sha256) for k in range(4) for r in endo_pick[k]]
sel += [dict(dataset="SAGES 2024", stratum=k, unit=r.clip_id, frame=int(r.frame_id), file=r.file, sha256=r.png_sha256) for k in range(4) for r in sages_pick[k]]
pd.DataFrame(sel).to_csv(os.path.join(OUT, "Fig3_frame_provenance.csv"), index=False)

# ---- 3. Fig. 9 trajectories + disagreement
g = L[L.video == 196].sort_values("frame").reset_index(drop=True)
t = (g.frame - g.frame.min()) / 25.0  # seconds (25 fps)
fig = plt.figure(figsize=(190 * MM, 150 * MM))
# thumbnails
picks = np.linspace(0, len(g) - 1, 8).round().astype(int)
th_axes = []
for i, idx in enumerate(picks):
    ax = fig.add_axes([0.075 + i * 0.113, 0.835, 0.105, 0.105])
    r = g.iloc[idx]; ax.imshow(endo_img(196, int(r.frame))); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(True); s_.set_edgecolor(POS if r.y_c1 else NEG); s_.set_linewidth(1.3)
    ax.set_title(f"{t[idx]:.0f} s", fontsize=6, pad=1.5); th_axes.append((ax, t[idx]))
fig.text(0.005, 0.975, "(a)", fontsize=8.5, fontweight="bold", va="top")
fig.text(0.075, 0.965, "Endoscapes operation 196, 75 annotated frames", fontsize=6.8, va="top", fontweight="bold")
for k_, (lab_, col_) in enumerate((("C1 positive", POS), ("C1 negative", NEG))):
    fig.patches.append(Rectangle((0.70 + k_ * 0.13, 0.952), 0.014, 0.012, transform=fig.transFigure, fc="white", ec=col_, lw=1.3))
    fig.text(0.718 + k_ * 0.13, 0.958, lab_, fontsize=6.3, va="center")
# label track
axl = fig.add_axes([0.075, 0.735, 0.895, 0.06])
for ci, c in enumerate(("c1", "c2", "c3")):
    for i in range(len(g)):
        v = g[f"official_{c}"].iloc[i]
        axl.add_patch(Rectangle((t[i] - 2.5, 2 - ci), 5, 0.85, fc=plt.cm.Oranges(0.15 + 0.75 * v), ec="none"))
axl.set_xlim(t.min() - 3, t.max() + 3); axl.set_ylim(0, 3); axl.set_yticks([2.42, 1.42, 0.42]); axl.set_yticklabels(["C1", "C2", "C3"])
axl.set_xticks([]); [axl.spines[k].set_visible(False) for k in ("left", "bottom")]; axl.tick_params(length=0)
for k_ in range(4):
    fig.patches.append(Rectangle((0.52 + k_ * 0.03, 0.952), 0.028, 0.012, transform=fig.transFigure, fc=plt.cm.Oranges(0.15 + 0.75 * k_ / 3), ec="none"))
    fig.text(0.534 + k_ * 0.03, 0.968, f"{k_}/3", fontsize=6, ha="center", va="bottom")
fig.text(0.515, 0.958, "C1\u2013C3 votes", fontsize=6.3, ha="right", va="center")
for ax, tt in th_axes:
    fig.add_artist(ConnectionPatch(xyA=(0.5, 0.0), coordsA=ax.transAxes, xyB=(tt, 3.0), coordsB=axl.transData, color="#9AA5B4", lw=0.5))
# score curves
axs = fig.add_axes([0.075, 0.47, 0.895, 0.24])
for i in range(len(g)):
    if g.y_c1.iloc[i]: axs.axvspan(t[i] - 2.5, t[i] + 2.5, color=POS, alpha=0.10, lw=0)
for s in (17, 29, 43):
    axs.plot(t, g[f"p_c1_seed{s}"], ls=SEEDS[s]["ls"], marker=SEEDS[s]["marker"], ms=2.0, lw=0.9, color=SEEDS[s]["color"], label=f"seed {s}")
for idx_f in (27400, 35025):
    i = int(np.flatnonzero(g.frame == idx_f)[0])
    axs.annotate("N" + ("1" if idx_f == 27400 else "2"), (t[i], g.p_c1_seed17.iloc[i]), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=6, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#000000", shrinkA=0, shrinkB=2))
axs.set_xlim(t.min() - 3, t.max() + 3); axs.set_ylim(-0.03, 1.05); axs.set_ylabel("C1 score"); axs.set_xlabel("Time from first annotated frame (s)")
axs.legend(loc="upper center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, -0.235))
from scipy.stats import rankdata
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
txt = "; ".join(f"seed {s}: within-operation AUROC {auc(g.y_c1.values, g[f'p_c1_seed{s}'].values):.3f}" for s in (17, 29, 43))
print("FIG9_AUROC", txt)
# disagreement scatter (b-d)
P = pd.read_csv(PAIRS); P["criterion"] = P.criterion.str.upper()
for j, c in enumerate(("C1", "C2", "C3")):
    ax = fig.add_axes([0.075 + j * 0.31, 0.07, 0.275, 0.27])
    q = P[P.criterion == c]; pooled = q.opposite_order_pairs.sum() / q.within_pairs.sum()
    ax.scatter(q.within_pairs, q.opposite_fraction * 100, s=10, color="#333333", edgecolor="white", lw=0.3, zorder=3)
    if c == "C1":
        r = q[q.video == 196].iloc[0]; ax.scatter(r.within_pairs, r.opposite_fraction * 100, s=30, facecolor="none", edgecolor="#000000", lw=0.9, zorder=4)
        ax.annotate("op. 196", (r.within_pairs, r.opposite_fraction * 100), xytext=(4, -9), textcoords="offset points", fontsize=6, color="#000000")
    ax.axhline(pooled * 100, color="#333333", lw=0.8, ls="--"); ax.axhline(q.opposite_fraction.median() * 100, color="#888888", lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_xlim(4, 1e4); ax.set_ylim(-2, 60)
    ax.set_title(f"{c}: {len(q)} operations, {q.within_pairs.sum():,} pairs", fontsize=6.8)
    ax.text(9.5e3, 58.5, f"pair-weighted {pooled*100:.1f}%", ha="right", va="top", fontsize=6, color="#333333")
    ax.text(9.5e3, 54.5, f"median {q.opposite_fraction.median()*100:.1f}%", ha="right", va="top", fontsize=6, color="#666666")
    ax.set_xlabel("Within-operation positive–negative pairs")
    if j == 0:
        ax.set_ylabel("Pairs ordered oppositely\nby seeds 17 and 29 (%)")
        ax.legend(handles=[Line2D([], [], color="#333333", lw=0.8, ls="--"), Line2D([], [], color="#888888", lw=0.9, ls=":")],
                  labels=["pair-weighted", "median"], loc="upper left", frameon=False, fontsize=6, handlelength=2.2)
    else: ax.set_yticklabels([])
    ax.yaxis.grid(True, lw=0.4, color="#E8E8E8"); ax.set_axisbelow(True)
    ax.text(-0.10 if j else -0.22, 1.10, f"({'bcd'[j]})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")
save(fig, "Fig9_trajectories_disagreement")
print("ok")
