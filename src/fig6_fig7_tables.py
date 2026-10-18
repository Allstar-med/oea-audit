"""Fig. 6, Fig. 7 and Tables 5, 6 and the comparison and country supplementary tables.
Usage: python fig6_fig7_tables.py DATA_DIR REANALYSIS_DIR FIG_DIR TABLE_DIR"""
import sys
import numpy as np, pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from decimal import Decimal, ROUND_HALF_UP

DATA, RES, FG, TB = sys.argv[1:5]
MM = 1 / 25.4
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.3, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.frameon": False, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
C = dict(E1="#000000", W="#009E73", E2="#CC79A7", pos="#D55E00", grey="#999999")


def h(v, n=3):
    return str(Decimal(repr(float(v))).quantize(Decimal(10) ** -n, rounding=ROUND_HALF_UP))


sg = lambda v, n=3: ("0." + "0" * n) if float(h(abs(v), n)) == 0 else ("+" if v >= 0 else "−") + h(abs(v), n)
ci = lambda lo, hi: f"[{sg(lo)}, {sg(hi)}]"
cip = lambda lo, hi: f"[{h(lo)}, {h(hi)}]"


def pipe(df, al):
    sep = ["---" if a == "l" else "--:" for a in al]
    return "\n".join(["| " + " | ".join(df.columns) + " |", "| " + " | ".join(sep) + " |"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in df.values])


# data
E = pd.read_csv(f"{DATA}/endoscapes/Endoscapes_10000_operation_bootstrap_CI.csv"); E["criterion"] = E.criterion.str.upper()
EC = pd.read_csv(f"{DATA}/endoscapes/Endoscapes_10000_paired_contrast_CI.csv"); EC["criterion"] = EC.criterion.str.upper()
S = pd.read_csv(f"{DATA}/frozen/Tables/SAGES_10000_bootstrap.csv"); S["criterion"] = S.criterion.str.upper(); S["seed"] = S.model.str.extract(r"seed(\d+)").astype(int)
ST = pd.read_csv(f"{DATA}/sages/results/source_test_estimands.csv")
IC = pd.read_csv(f"{RES}/model_comparisons.csv")
MC = pd.read_csv(f"{DATA}/sages/results/model_comparisons.csv"); MC = MC[MC.analysis == "primary"].copy()
MC["a"] = MC.model_a.str.extract(r"seed(\d+)").astype(int); MC["b"] = MC.model_b.str.extract(r"seed(\d+)").astype(int)
CB = pd.read_csv(f"{RES}/country_bootstrap_W_minus_E1.csv"); CB["seed"] = CB.model.str.extract(r"seed(\d+)").astype(int)
crit = ["C1", "C2", "C3"]; seeds = [17, 29, 43]


def endo(c, s, est):
    r = E[(E.criterion == c) & (E.seed == s) & (E.estimand == est)].iloc[0]; return r.point, r.ci_low, r.ci_high


def endoc(c, s, con):
    r = EC[(EC.criterion == c) & (EC.seed == s) & (EC.contrast == con)].iloc[0]; return r.point, r.ci_low, r.ci_high


# Table 5: both datasets with CIs
rows = []
for ds in ("Endoscapes", "SAGES"):
    for c in crit:
        for s in seeds:
            if ds == "Endoscapes":
                e1, w, e2 = endo(c, s, "E1"), endo(c, s, "W"), endo(c, s, "E2"); d1, d2 = endoc(c, s, "W-E1"), endoc(c, s, "E2-E1")
            else:
                r = S[(S.criterion == c) & (S.seed == s)].iloc[0]
                e1, w, e2 = (r.E1, r.E1_lo, r.E1_hi), (r.W, r.W_lo, r.W_hi), (r.E2, r.E2_lo, r.E2_hi)
                d1, d2 = (r.W_minus_E1, r.W_minus_E1_lo, r.W_minus_E1_hi), (r.E2_minus_E1, r.E2_minus_E1_lo, r.E2_minus_E1_hi)
            rows.append(["Endo." if ds == "Endoscapes" else "SAGES", c, s, f"{h(e1[0])} {cip(*e1[1:])}", f"{h(w[0])} {cip(*w[1:])}",
                         f"{sg(d1[0])} {ci(*d1[1:])}", f"{sg(d2[0])} {ci(*d2[1:])}"])
T5 = pd.DataFrame(rows, columns=["Data", "Crit.", "Seed", "$E_1$ [95% CI]", "$W$ [95% CI]", "$W-E_1$ [95% CI]", "$E_2-E_1$ [95% CI]"])
open(f"{TB}/T5.md", "w").write(pipe(T5, ["l", "l", "r"] + ["r"] * 4) + "\n")

# Table 6: same checkpoints on both datasets
rows = []
for c in crit:
    for s in seeds:
        r = ST[(ST.criterion == c) & (ST.seed == s)].iloc[0]; q = S[(S.criterion == c) & (S.seed == s)].iloc[0]
        de, dsg = r.W - r.E1, q.W_minus_E1
        rows.append([c, s, f"{h(r.E1)} {cip(r.E1_lo, r.E1_hi)}", h(r.W), f"{h(r.E2)} {cip(r.E2_lo, r.E2_hi)}", sg(de),
                     f"{sg(dsg)} {ci(q.W_minus_E1_lo, q.W_minus_E1_hi)}", "yes" if np.sign(de) == np.sign(dsg) else "no"])
T6 = pd.DataFrame(rows, columns=["Crit.", "Seed", "Endo. $E_1$ [95% CI]", "Endo. $W$", "Endo. $E_2$ [95% CI]",
                                 "Endo. $W-E_1$", "SAGES $W-E_1$ [95% CI]", "Same sign"])
open(f"{TB}/T6.md", "w").write(pipe(T6, ["l", "r"] + ["r"] * 6) + "\n")
agree = (T6["Same sign"] == "yes").sum(); print("same-sign", agree)

# supplementary table: model comparisons, both datasets
rows = []
for est in ("E1", "W", "E2"):
    for c in crit:
        for a, b in ((17, 29), (17, 43), (29, 43)):
            r = MC[(MC.criterion == c) & (MC.estimand == est) & (MC.a == a) & (MC.b == b)].iloc[0]
            i = IC[(IC.dataset == "Endoscapes") & (IC.criterion == c) & (IC.estimand == est) & (IC.model_a == f"seed{a}") & (IC.model_b == f"seed{b}")].iloc[0]
            j = IC[(IC.dataset == "SAGES 2024") & (IC.criterion == c) & (IC.estimand == est) & (IC.model_a == f"seed{a}") & (IC.model_b == f"seed{b}")].iloc[0]
            lab = {"E1": "$E_1$", "W": "$W$", "E2": "$E_2$"}[est]
            rows.append([lab, c, f"{a} vs {b}", f"{sg(i.difference)} {ci(i.ci_lo, i.ci_hi)}", h(i.p_two_sided, 4), h(i.q_BH, 4),
                         f"{sg(r.diff_a_minus_b)} {ci(r.lo, r.hi)}", h(r.p_boot_two_sided, 4), h(r.q_BH_within_family, 4),
                         h(j.p_two_sided, 4), h(j.q_BH, 4)])
S3 = pd.DataFrame(rows)
ta = S3[[0, 1, 2, 3, 4, 5]].copy(); ta.columns = ["Estimand", "Crit.", "Seeds", "Difference [95% CI]", "$p$", "$q$"]
tb = S3[[0, 1, 2, 6, 7, 8, 9, 10]].copy(); tb.columns = ["Estimand", "Crit.", "Seeds", "Difference [95% CI]", "$p$", "$q$", "$p$ (recomputed)", "$q$ (recomputed)"]
open(f"{TB}/S1.md", "w").write("*(a) Endoscapes*\n\n" + pipe(ta, ["l", "l", "l", "r", "r", "r"]) + "\n\n*(b) SAGES 2024*\n\n" + pipe(tb, ["l", "l", "l", "r", "r", "r", "r", "r"]) + "\n")
# summary numbers
endo_sig = IC[(IC.dataset == "Endoscapes") & (IC.q_BH < 0.05)]
print("Endoscapes BH sig:\n", endo_sig[["criterion", "estimand", "model_a", "model_b", "difference", "ci_lo", "ci_hi", "p_two_sided", "q_BH"]].round(4).to_string())
print("Endoscapes unadjusted CI excl 0:", ((IC.dataset == "Endoscapes") & ((IC.ci_lo > 0) | (IC.ci_hi < 0))).sum())
print("SAGES replication BH sig:", IC[(IC.dataset == "SAGES 2024") & (IC.q_BH < 0.05)][["criterion", "estimand", "model_a", "model_b", "q_BH"]].round(4).to_string())

# supplementary table: country-level resampling
rows = []
for c in crit:
    for s in seeds:
        r = CB[(CB.criterion == c) & (CB.seed == s)].iloc[0]; q = S[(S.criterion == c) & (S.seed == s)].iloc[0]
        rows.append([c, s, sg(r.W_minus_E1), ci(q.W_minus_E1_lo, q.W_minus_E1_hi), ci(r.lo, r.hi)])
S6 = pd.DataFrame(rows, columns=["Criterion", "Seed", "$W-E_1$", "95% CI, clips resampled (293)", "95% CI, countries resampled (18)"])
open(f"{TB}/S6.md", "w").write(pipe(S6, ["l", "r", "r", "r", "r"]) + "\n")


# Fig. 6
UMINUS = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace("-", "\u2212") if abs(v) > 1e-9 else "0.00")
def save(fig, name):
    for ext in ("pdf", "png", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(f"{FG}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300 if ext == "png" else 600, **kw)
    plt.close(fig)


fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 80 * MM), sharey=True, gridspec_kw=dict(wspace=0.10))
ypos, labels = [], []
y = 0
for c in crit:
    for s in seeds:
        ypos.append(y); labels.append(f"{c}  seed {s}"); y -= 1
    y -= 0.6
k = 0
for c in crit:
    for s in seeds:
        yy = ypos[k]; k += 1
        for j, (con, col, dy) in enumerate((("W-E1", C["W"], 0.14), ("E2-E1", C["E2"], -0.14))):
            p, lo, hi = endoc(c, s, con)
            exc = lo > 0 or hi < 0
            axs[0].errorbar(p, yy + dy, xerr=[[p - lo], [hi - p]], fmt="o", ms=2.8, color=col, mfc=col if exc else "white", elinewidth=0.8)
            r = S[(S.criterion == c) & (S.seed == s)].iloc[0]
            key = "W_minus_E1" if con == "W-E1" else "E2_minus_E1"
            p, lo, hi = r[key], r[key + "_lo"], r[key + "_hi"]; exc = lo > 0 or hi < 0
            axs[1].errorbar(p, yy + dy, xerr=[[p - lo], [hi - p]], fmt="o", ms=2.8, color=col, mfc=col if exc else "white", elinewidth=0.8)
        st = ST[(ST.criterion == c) & (ST.seed == s)].iloc[0]; q = S[(S.criterion == c) & (S.seed == s)].iloc[0]
        axs[2].plot([st.W - st.E1, q.W_minus_E1], [yy + 0.14, yy - 0.14], color="#BBBBBB", lw=0.6, zorder=1)
        axs[2].plot(st.W - st.E1, yy + 0.14, "D", ms=3, color="#444444", zorder=3)
        exc = q.W_minus_E1_lo > 0 or q.W_minus_E1_hi < 0
        axs[2].errorbar(q.W_minus_E1, yy - 0.14, xerr=[[q.W_minus_E1 - q.W_minus_E1_lo], [q.W_minus_E1_hi - q.W_minus_E1]], fmt="o", ms=2.8,
                        color=C["W"], mfc=C["W"] if exc else "white", elinewidth=0.8, zorder=2)
for ax, L, t in zip(axs, "abc", ["Endoscapes, original checkpoints\n(operations resampled)", "SAGES 2024, retrained checkpoints\n(clips resampled)", "Same retrained checkpoints\non both datasets"]):
    ax.axvline(0, color="#666666", lw=0.6); ax.xaxis.grid(True, lw=0.4, color="#E8E8E8"); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(UMINUS)
    ax.set_title(t, fontsize=7); ax.text(-0.04 if L != "a" else -0.30, 1.08, f"({L})", transform=ax.transAxes, fontsize=8, fontweight="bold")
xl = [min(a.get_xlim()[0] for a in axs), max(a.get_xlim()[1] for a in axs)]
for a in axs: a.set_xlim(xl); a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
axs[0].set_yticks(ypos); axs[0].set_yticklabels(labels)
axs[0].set_xlabel("Difference from $E_1$ (95% CI)"); axs[1].set_xlabel("Difference from $E_1$ (95% CI)")
axs[2].set_xlabel("$W-E_1$")
axs[0].legend(handles=[Line2D([], [], marker="o", ls="", color=C["W"], ms=3), Line2D([], [], marker="o", ls="", color=C["E2"], ms=3),
                       Line2D([], [], marker="o", ls="", color="#666666", mfc="white", ms=3)], labels=["$W-E_1$", "$E_2-E_1$", "CI includes 0"],
              loc="upper center", bbox_to_anchor=(1.05, -0.16), ncol=3)
axs[2].legend(handles=[Line2D([], [], marker="D", ls="", color="#444444", ms=3), Line2D([], [], marker="o", ls="-", color=C["W"], ms=3, lw=0.8)],
              labels=["Endoscapes (point)", "SAGES (95% CI)"], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
save(fig, "Fig6_estimates_contrasts")

# Fig. 7
QL = []
fig, axs = plt.subplots(2, 3, figsize=(190 * MM, 110 * MM), sharey=True, sharex=True, gridspec_kw=dict(wspace=0.08, hspace=0.42))
pairs = [(17, 29), (17, 43), (29, 43)]
yl, yp = [], []; y = 0
for c in crit:
    for p in pairs:
        yl.append(f"{c}  {p[0]} vs {p[1]}"); yp.append(y); y -= 1
    y -= 0.5
for r_, ds in enumerate(("Endoscapes", "SAGES 2024")):
    for c_, est in enumerate(("E1", "W", "E2")):
        ax = axs[r_, c_]; k = 0
        for c in crit:
            for a, b in pairs:
                if ds == "Endoscapes":
                    i = IC[(IC.dataset == ds) & (IC.criterion == c) & (IC.estimand == est) & (IC.model_a == f"seed{a}") & (IC.model_b == f"seed{b}")].iloc[0]
                    d, lo, hi, q = i.difference, i.ci_lo, i.ci_hi, i.q_BH
                else:
                    i = MC[(MC.criterion == c) & (MC.estimand == est) & (MC.a == a) & (MC.b == b)].iloc[0]
                    d, lo, hi, q = i.diff_a_minus_b, i.lo, i.hi, i.q_BH_within_family
                unadj = lo > 0 or hi < 0
                ax.errorbar(d, yp[k], xerr=[[d - lo], [hi - d]], fmt="o", ms=2.8, color=C[est], mfc=C[est] if unadj else "white", elinewidth=0.8)
                if q < 0.05:
                    QL.append((ax, yp[k], q))
                k += 1
        ax.axvline(0, color="#666666", lw=0.6); ax.xaxis.grid(True, lw=0.4, color="#E8E8E8"); ax.set_axisbelow(True)
        ax.tick_params(labelbottom=True); ax.xaxis.set_major_formatter(UMINUS)
        ax.set_title(f"{ds.replace(' 2024', '')}: " + {"E1": "$E_1$", "W": "$W$", "E2": "$E_2$"}[est], fontsize=7)
        L = "abcdef"[r_ * 3 + c_]
        ax.text(-0.04 if c_ else -0.42, 1.04, f"({L})", transform=ax.transAxes, fontsize=8, fontweight="bold")
    axs[r_, 0].set_yticks(yp); axs[r_, 0].set_yticklabels(yl)
lo_, hi_ = axs[0, 0].get_xlim(); axs[0, 0].set_xlim(lo_, hi_ + 0.45 * (hi_ - lo_))
axs[0, 0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-0.1, 0, 0.1, 0.2]))
for ax, yy, q in QL:
    ax.text(0.98, yy, f"q = {q:.3f}", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData), ha="right", va="center", fontsize=6)
axs[1, 1].set_xlabel("Paired difference in AUROC (first minus second seed), 95% CI")
axs[1, 2].legend(handles=[Line2D([], [], marker="o", ls="", color="#444444", ms=3), Line2D([], [], marker="o", ls="", color="#444444", mfc="white", ms=3)],
                 labels=["unadjusted CI excludes 0", "CI includes 0"], loc="upper center", bbox_to_anchor=(-0.55, -0.28), ncol=2)
save(fig, "Fig7_model_comparisons")
print("done")
