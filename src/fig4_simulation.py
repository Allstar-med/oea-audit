"""Fig. 4 and the simulation summary tables.
Usage: python fig4_simulation.py SIM_DIR FIG_DIR TABLE_DIR"""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM, FIG, TAB = sys.argv[1:4]
MM = 1 / 25.4
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
a = pd.read_csv(f"{SIM}/sim_A_raw.csv"); a["rho"] = a.rho.round(1); a["d"] = a.W - a.E1
g = a.groupby(["sigma_b", "rho"])
sa = g.agg(mean=("d", "mean"), lo=("d", lambda v: v.quantile(.025)), hi=("d", lambda v: v.quantile(.975)),
           p_pos=("d", lambda v: (v > 0).mean()), W=("W", "mean"), E1=("E1", "mean"), E2W=("E2", "mean"),
           alpha=("alpha", "mean"), elig=("eligible", "mean")).reset_index()
sa["E2W"] = sa.E2W - sa.W
sa.to_csv(f"{TAB}/sim_A_summary.csv", index=False)
truth = json.load(open(f"{SIM}/sim_truth.json"))
b = pd.read_json(f"{SIM}/sim_B_raw.json"); b["rho"] = b.rho.round(1)
rows = []
for (sb, rho), gg in b.groupby(["sigma_b", "rho"]):
    te = truth[f"{float(sb)}_{float(rho)}"]; tw = truth["W"]
    def cov(col, tv):
        x = np.array(gg[col].tolist()); hit = (x[:, 0] <= tv) & (tv <= x[:, 1]); return hit.mean(), (x[:, 1] - x[:, 0]).mean()
    for lab, col, tv in [("E1, frame resampling", "E1_frame", te), ("E1, unit resampling", "E1_clu", te), ("W, unit resampling", "W_clu", tw)]:
        c, w = cov(col, tv)
        rows.append(dict(sigma_b=sb, rho=rho, interval=lab, coverage=c, mc_se=np.sqrt(c * (1 - c) / len(gg)), mean_width=w,
                         n_rep=len(gg), true_value=tv))
sb_ = pd.DataFrame(rows); sb_.to_csv(f"{TAB}/sim_B_summary.csv", index=False)

fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 62 * MM), gridspec_kw=dict(wspace=0.42, width_ratios=[1.15, 1, 0.8]))
ax = axs[0]
cols = {0.0: "#9ECAE1", 0.5: "#4292C6", 1.0: "#08519C", 2.0: "#08214A"}
for s, gg in sa.groupby("sigma_b"):
    if s == 1.0: ax.fill_between(gg.rho, gg.lo, gg.hi, color=cols[s], alpha=0.15, lw=0)
    ax.plot(gg.rho, gg["mean"], "-o", color=cols[s], ms=2.6, lw=1.0, label=f"$\\sigma_b$ = {s:g}", zorder=3)
ax.axhline(0, color="#999999", lw=0.6, ls=":", zorder=1)
ax.set_xlabel("Correlation $\\rho$ of unit offset with $z_u$")
ax.set_ylabel("$W-E_1$ (AUROC difference)")
ax.legend(loc="upper right", ncol=2)
ax.text(-0.13, 1.10, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold")
ax.yaxis.grid(True, lw=0.4, color="#E8E8E8"); ax.set_axisbelow(True)
ax = axs[1]
scen = [(0.0, 0.0), (1.0, -0.6), (1.0, 0.0), (1.0, 0.6)]
lab = ["$\\sigma_b$ = 0", "$\\sigma_b$ = 1\n$\\rho$ = −0.6", "$\\sigma_b$ = 1\n$\\rho$ = 0", "$\\sigma_b$ = 1\n$\\rho$ = 0.6"]
kinds = [("E1, frame resampling", "#C7C7C7"), ("E1, unit resampling", "#000000"), ("W, unit resampling", "#009E73")]
nice = {"E1, frame resampling": "$E_1$, frames", "E1, unit resampling": "$E_1$, units", "W, unit resampling": "$W$, units"}
bw = 0.26
for k, (kind, col) in enumerate(kinds):
    for i, (s, r) in enumerate(scen):
        row = sb_[(sb_.sigma_b == s) & (sb_.rho == r) & (sb_.interval == kind)].iloc[0]
        ax.bar(i + (k - 1) * bw, row.coverage, bw, color=col, label=nice[kind] if i == 0 else None)
        ax.errorbar(i + (k - 1) * bw, row.coverage, yerr=1.96 * row.mc_se, color="#333333", lw=0.6, capsize=0)
ax.axhline(0.95, color="#333333", lw=0.6, ls="--")
ax.set_ylim(0, 1.0); ax.set_xticks(range(4)); ax.set_xticklabels(lab)
ax.set_ylabel("Coverage of nominal 95% interval")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, handlelength=1.0, columnspacing=0.8)
ax.text(-0.17, 1.10, "(b)", transform=ax.transAxes, fontsize=8, fontweight="bold")
# (c) coverage by effective number of units
ax = axs[2]
bn = pd.read_csv(f"{SIM}/sim_C_by_neff.csv"); bn = bn[bn.reps >= 100].reset_index(drop=True)
xc = np.arange(len(bn))
for col, lab, mk, fc in (("W_cover", "$W$, percentile", "o", "#009E73"), ("Wt_cover", "$W$, $t$ with $n_{\\mathrm{eff}}-1$ df", "s", "white"),
                         ("E2_cover", "$E_2$, percentile", "^", "#CC79A7")):
    se = np.sqrt(bn[col] * (1 - bn[col]) / bn.reps)
    ax.errorbar(xc + {"W_cover": -0.15, "Wt_cover": 0.0, "E2_cover": 0.15}[col], bn[col], yerr=1.96 * se, fmt=mk, ms=3.6, color="#009E73" if col != "E2_cover" else "#CC79A7", mfc=fc, lw=0.7, label=lab)
ax.axhline(0.95, color="#333333", lw=0.6, ls="--")
ax.set_xticks(xc); ax.set_xticklabels([s.replace("-", "\u2013").replace(">=", "\u2265") for s in bn.neff_bin]); ax.set_xlabel("Effective number of units, $n_{\\mathrm{eff}}$")
ax.set_ylim(0.85, 0.98); ax.set_ylabel("Coverage of nominal 95% interval")
ax.legend(loc="lower right", fontsize=6, handletextpad=0.3)
ax.text(-0.30, 1.10, "(c)", transform=ax.transAxes, fontsize=8, fontweight="bold")
for ext in ("pdf", "png", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(f"{FIG}/Fig4_simulation.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300 if ext == "png" else 600, **kw)
print(sa[sa.sigma_b.isin([0, 1.0])].round(3).to_string())
print(sb_.round(3).to_string()); print(truth)
