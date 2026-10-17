"""Operation-aware estimand audit for frame-level AUROC on clustered video data.

Usage (row-level file with one row per frame and model):
    python oea_audit.py --csv predictions.csv --unit unit_id --label y \
        --score score --model model_id --n-boot 10000 --seed 20261010 --out results/

Computes, for each model: E1 (pooled AUROC), X (cross-unit AUROC), W (within-unit
pair-weighted AUROC), E2 (equal-weight mean of within-unit AUROC over mixed-label
units), alpha (within-unit pair fraction), the number of mixed-label units, the Kish
effective number of units behind W, and percentile cluster-bootstrap intervals in
which units are resampled intact and the same resample indices are used for every
model, so that paired model contrasts are available.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auc(y, s):
    """Mann-Whitney AUROC with ties counted as 1/2."""
    y = np.asarray(y, dtype=bool)
    n1 = y.sum(); n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def unit_summaries(ys, ss):
    """Per-unit positive count, negative count and within-unit AUROC."""
    npos = np.array([y.sum() for y in ys], dtype=float)
    nneg = np.array([y.size - y.sum() for y in ys], dtype=float)
    a = np.array([auc(y, s) if (p > 0 and n > 0) else np.nan
                  for y, s, p, n in zip(ys, ss, npos, nneg)])
    return npos, nneg, a


def estimands(ys, ss, idx=None, cache=None):
    if idx is None:
        idx = np.arange(len(ys))
    y = np.concatenate([ys[i] for i in idx]); s = np.concatenate([ss[i] for i in idx])
    e1 = auc(y, s)
    npos, nneg, a = cache if cache is not None else unit_summaries(ys, ss)
    w = (npos * nneg)[idx]; au = a[idx]
    ok = w > 0
    W = np.sum(w[ok] * au[ok]) / np.sum(w[ok]) if ok.any() else np.nan
    E2 = np.mean(au[ok]) if ok.any() else np.nan
    N = y.sum() * (y.size - y.sum())
    alpha = w[ok].sum() / N if N > 0 else np.nan
    X = (e1 - alpha * W) / (1 - alpha) if alpha < 1 else np.nan
    neff = w[ok].sum() ** 2 / np.sum(w[ok] ** 2) if ok.any() else np.nan
    return dict(E1=e1, X=X, W=W, E2=E2, alpha=alpha, eligible=int(ok.sum()), neff=neff,
                W_minus_E1=W - e1, E2_minus_E1=E2 - e1, E2_minus_W=E2 - W)


def run(df, unit, label, score, model, n_boot, seed):
    units = np.sort(df[unit].unique())
    models = list(pd.unique(df[model]))
    data = {}
    for m in models:
        g = df[df[model] == m].set_index(unit)
        ys = [g.loc[[u], label].to_numpy(dtype=int) for u in units]
        ss = [g.loc[[u], score].to_numpy(dtype=float) for u in units]
        data[m] = (ys, ss, unit_summaries(ys, ss))
    point = {m: estimands(*data[m][:2], cache=data[m][2]) for m in models}
    rng = np.random.default_rng(seed)
    keys = ["E1", "X", "W", "E2", "W_minus_E1", "E2_minus_E1", "E2_minus_W"]
    draws = {m: [] for m in models}
    for _ in range(n_boot):
        idx = rng.integers(0, len(units), len(units))   # same indices for all models
        for m in models:
            r = estimands(*data[m][:2], idx=idx, cache=data[m][2])
            draws[m].append([r[k] for k in keys])
    rows = []
    for m in models:
        d = np.array(draws[m], dtype=float)
        row = dict(model=m, **point[m])
        for j, k in enumerate(keys):
            row[k + "_lo"], row[k + "_hi"] = np.nanquantile(d[:, j], [0.025, 0.975])
        rows.append(row)
    comps = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            da = np.array(draws[a], dtype=float); db = np.array(draws[b], dtype=float)
            for k_i, k in enumerate(["E1", "W", "E2"]):
                diff = da[:, keys.index(k)] - db[:, keys.index(k)]
                diff = diff[~np.isnan(diff)]
                obs = point[a][k] - point[b][k]
                p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
                lo, hi = np.quantile(diff, [0.025, 0.975])
                comps.append(dict(estimand=k, model_a=a, model_b=b, difference=obs,
                                  ci_lo=lo, ci_hi=hi, p_two_sided=p))
    comps = pd.DataFrame(comps)
    if len(comps):
        comps["q_BH_within_estimand"] = comps.groupby("estimand").p_two_sided.transform(bh)
    return pd.DataFrame(rows), comps


def bh(p):
    p = np.asarray(p, dtype=float); n = p.size; o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for r in range(n - 1, -1, -1):
        prev = min(prev, p[o[r]] * n / (r + 1)); q[o[r]] = prev
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--unit", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--score", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261010); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    est, comp = run(pd.read_csv(a.csv), a.unit, a.label, a.score, a.model, a.n_boot, a.seed)
    est.to_csv(f"{a.out}/estimands_with_ci.csv", index=False)
    comp.to_csv(f"{a.out}/paired_model_comparisons.csv", index=False)
    print(est.round(4).to_string(index=False))
