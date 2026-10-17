"""Experiment C: coverage of unit-level percentile bootstrap intervals for W and E2 when the test-set design
(unit sizes and label sequences) is resampled from the real Endoscapes or SAGES 2024 test sets.
Each replicate draws U units with replacement from the real designs (keeping each unit's frame labels),
generates scores s = delta_u * y + b_u + e with delta_u ~ N(0.8, 0.4^2), b_u ~ N(0, 1), e ~ N(0, 1),
and computes W and E2 with B unit-level bootstrap resamples. Because E[A_u | delta_u] = Phi(delta_u / sqrt(2))
and delta_u is independent of the design, the population W and E2 both equal E[Phi(delta / sqrt(2))].
A Student-t interval for W (bootstrap standard error, n_eff - 1 degrees of freedom) is evaluated alongside.
python simulation_design.py <data_dir> <out_dir> [replicates] [B]"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata, t as tdist

D, OUT = sys.argv[1:3]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
B = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261017)
truth = norm.cdf(rng.normal(0.8, 0.4, 2_000_000) / np.sqrt(2)).mean()


def designs():
    L = pd.read_csv(f"{D}/endoscapes/endoscapes_test_linkage.csv")
    S = pd.read_csv(f"{D}/sages/sages_labels.csv"); S = S[S.primary_include == 1]
    out = {}
    for c in (1, 2, 3):
        out[("Endoscapes", f"C{c}")] = [g[f"y_c{c}"].astype(int).values for _, g in L.groupby("video")]
        out[("SAGES 2024", f"C{c}")] = [g[f"c{c}_majority"].astype(int).values for _, g in S.groupby("clip_id")]
    return out


def unit_auc(p, n):
    r = rankdata(np.r_[p, n])
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


rows = []
for (ds, c), units in designs().items():
    U = len(units)
    npos = np.array([u.sum() for u in units]); nneg = np.array([len(u) - u.sum() for u in units])
    for r in range(R):
        pick = rng.integers(0, U, U)
        a = np.full(U, np.nan); w = (npos[pick] * nneg[pick]).astype(float)
        for k, j in enumerate(pick):
            if w[k] == 0:
                continue
            y = units[j]; dlt = rng.normal(0.8, 0.4); b = rng.normal()
            s = dlt * y + b + rng.normal(size=len(y))
            a[k] = unit_auc(s[y == 1], s[y == 0])
        mixed = w > 0
        if mixed.sum() < 2:
            continue
        a0 = np.nan_to_num(a)
        W = (w * a0).sum() / w.sum(); E2 = a0[mixed].mean(); neff = w.sum() ** 2 / (w ** 2).sum()
        dr = rng.integers(0, U, size=(B, U)); wd = w[dr]; md = mixed[dr].astype(float); ad = a0[dr]
        ok = wd.sum(1) > 0
        Wb = (wd * ad).sum(1)[ok] / wd.sum(1)[ok]; E2b = (md * ad).sum(1)[ok] / md.sum(1)[ok]
        wl, wh = np.quantile(Wb, [0.025, 0.975]); el, eh = np.quantile(E2b, [0.025, 0.975])
        q = tdist.ppf(0.975, max(neff - 1, 1)); se = Wb.std(ddof=1)
        tl, th = W - q * se, W + q * se
        rows.append(dict(dataset=ds, criterion=c, rep=r, mixed=int(mixed.sum()), neff=neff, W=W, E2=E2,
                         W_cover=wl <= truth <= wh, W_width=wh - wl, E2_cover=el <= truth <= eh, E2_width=eh - el,
                         Wt_cover=tl <= truth <= th, Wt_width=th - tl))
raw = pd.DataFrame(rows); raw.to_csv(f"{OUT}/sim_C_raw.csv", index=False)
raw["neff_bin"] = pd.cut(raw.neff, [0, 5, 10, 20, 50, 1000], labels=["<5", "5-10", "10-20", "20-50", ">=50"])
summ = raw.groupby(["dataset", "criterion"]).agg(reps=("rep", "size"), mixed=("mixed", "mean"), neff=("neff", "median"),
                                                 W_cover=("W_cover", "mean"), W_width=("W_width", "mean"),
                                                 E2_cover=("E2_cover", "mean"), E2_width=("E2_width", "mean"),
                                                 W_bias=("W", lambda v: v.mean() - truth),
                                                 Wt_cover=("Wt_cover", "mean"), Wt_width=("Wt_width", "mean")).reset_index()
summ["truth"] = truth
summ.to_csv(f"{OUT}/sim_C_summary.csv", index=False)
bybin = raw.groupby("neff_bin", observed=True).agg(reps=("rep", "size"), W_cover=("W_cover", "mean"), Wt_cover=("Wt_cover", "mean"),
                                                   E2_cover=("E2_cover", "mean")).reset_index()
bybin.to_csv(f"{OUT}/sim_C_by_neff.csv", index=False)
print("truth", round(truth, 4)); print(summ.round(3).to_string()); print(bybin.round(3).to_string())
