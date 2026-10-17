"""Reference-anchored within-unit AUROC and transition-distance sensitivity.
For each dataset, criterion and seed: W and E2 of the model, of a time-only reference score, and their paired
differences (model minus reference) with 95% percentile intervals from B matched cluster-bootstrap resamples.
Also a Student-t interval for W with n_eff - 1 degrees of freedom, and W restricted to within-unit pairs whose frames are at least `gap` seconds apart.
python reference_anchored.py <data_dir> <out_dir> [B]"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import rankdata, t as tdist

D, OUT = sys.argv[1:3]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
os.makedirs(OUT, exist_ok=True)
RNG_SEED = 20261017
GAP_S = 15.0


def unit_auc(p, n):
    r = rankdata(np.r_[p, n])
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def pair_auc(tp, tn, sp, sn, gap):
    """within-unit AUROC over pairs with |t_pos - t_neg| >= gap; returns (auc, n_pairs)"""
    m = np.abs(tp[:, None] - tn[None, :]) >= gap
    if not m.any():
        return np.nan, 0
    d = sp[:, None] - sn[None, :]
    k = (d > 0) + 0.5 * (d == 0)
    return k[m].mean(), int(m.sum())


def load():
    L = pd.read_csv(f"{D}/endoscapes/endoscapes_test_linkage.csv")
    L["unit"] = L.video.astype(str)
    L["t"] = (L.frame - L.groupby("video").frame.transform("min")) / 25.0  # Endoscapes videos at 25 fps
    endo = {}
    for c in (1, 2, 3):
        endo[f"C{c}"] = (L[["unit", "t"]].assign(y=L[f"y_c{c}"].astype(int)),
                         {s: L[f"p_c{c}_seed{s}"].values for s in (17, 29, 43)})
    S = pd.read_csv(f"{D}/sages/sages_labels.csv"); S = S[S.primary_include == 1]
    P = pd.read_csv(f"{D}/sages/sages_predictions.csv")
    P["seed"] = P.model_id.str.extract(r"seed(\d+)").astype(int)
    sages = {}
    for c in (1, 2, 3):
        base = S[["clip_id", "frame_id", "timestamp", f"c{c}_majority"]].rename(columns={"clip_id": "unit", "timestamp": "t", f"c{c}_majority": "y"})
        sc = {}
        for s in (17, 29, 43):
            q = base.merge(P[P.seed == s][["clip_id", "frame_id", f"p_c{c}"]].rename(columns={"clip_id": "unit"}), on=["unit", "frame_id"], how="left", validate="one_to_one")
            assert q[f"p_c{c}"].notna().all()
            sc[s] = q[f"p_c{c}"].values
        sages[f"C{c}"] = (base[["unit", "t", "y"]].reset_index(drop=True), sc)
    return {"Endoscapes": endo, "SAGES 2024": sages}


rng = np.random.default_rng(RNG_SEED)
rows, gaprows = [], []
for ds, crit in load().items():
    for c, (df, scores) in crit.items():
        units = df.unit.unique(); U = len(units)
        idx = {u: np.flatnonzero(df.unit.values == u) for u in units}
        y = df.y.values; t = df.t.values
        w = np.array([(y[idx[u]] == 1).sum() * (y[idx[u]] == 0).sum() for u in units], float)
        mixed = w > 0
        def per_unit(s):
            a = np.full(U, np.nan)
            for k, u in enumerate(units):
                if mixed[k]:
                    ii = idx[u]; a[k] = unit_auc(s[ii][y[ii] == 1], s[ii][y[ii] == 0])
            return a
        a_time = per_unit(t)
        neff = w.sum() ** 2 / (w ** 2).sum(); tq = tdist.ppf(0.975, max(neff - 1, 1))
        draws = rng.integers(0, U, size=(B, U))
        wd = w[draws]; md = mixed[draws].astype(float)
        def stats(a):
            a0 = np.nan_to_num(a)
            ad = a0[draws]
            Wb = (wd * ad).sum(1) / wd.sum(1)
            E2b = (md * ad).sum(1) / md.sum(1)
            W = (w * a0).sum() / w.sum(); E2 = a0[mixed].mean()
            return W, E2, Wb, E2b
        Wt, E2t, Wtb, E2tb = stats(a_time)
        for s, sc in scores.items():
            a_m = per_unit(sc)
            Wm, E2m, Wmb, E2mb = stats(a_m)
            dW, dWb = Wm - Wt, Wmb - Wtb
            dE, dEb = E2m - E2t, E2mb - E2tb
            q = lambda v: np.quantile(v, [0.025, 0.975])
            p2 = lambda v: min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
            mq = tdist.ppf(0.975, max(int(mixed.sum()) - 1, 1))
            tW = (dW - tq * dWb.std(ddof=1), dW + tq * dWb.std(ddof=1)); tE = (dE - mq * dEb.std(ddof=1), dE + mq * dEb.std(ddof=1))
            rows.append(dict(dataset=ds, criterion=c, seed=s, units=U, mixed=int(mixed.sum()),
                             dW_tlo=tW[0], dW_thi=tW[1], dE2_tlo=tE[0], dE2_thi=tE[1],
                             W_model=Wm, W_time=Wt, dW=dW, dW_lo=q(dWb)[0], dW_hi=q(dWb)[1], dW_p=p2(dWb),
                             E2_model=E2m, E2_time=E2t, dE2=dE, dE2_lo=q(dEb)[0], dE2_hi=q(dEb)[1], dE2_p=p2(dEb),
                             W_model_lo=q(Wmb)[0], W_model_hi=q(Wmb)[1], W_time_lo=q(Wtb)[0], W_time_hi=q(Wtb)[1],
                             neff=neff, W_model_tlo=Wm - tq * Wmb.std(ddof=1), W_model_thi=Wm + tq * Wmb.std(ddof=1)))
            # transition-distance sensitivity
            num_m = num_t = npairs = 0.0
            for k, u in enumerate(units):
                if not mixed[k]:
                    continue
                ii = idx[u]; pp, nn = ii[y[ii] == 1], ii[y[ii] == 0]
                am, n1 = pair_auc(t[pp], t[nn], sc[pp], sc[nn], GAP_S)
                at, _ = pair_auc(t[pp], t[nn], t[pp], t[nn], GAP_S)
                if n1:
                    num_m += am * n1; num_t += at * n1; npairs += n1
            # pairs that time orders incorrectly (positive frame earlier than negative frame)
            num_d = nd = 0.0
            for k, u in enumerate(units):
                if not mixed[k]:
                    continue
                ii = idx[u]; pp, nn = ii[y[ii] == 1], ii[y[ii] == 0]
                m = t[pp][:, None] < t[nn][None, :]
                if m.any():
                    d = sc[pp][:, None] - sc[nn][None, :]
                    num_d += ((d > 0) + 0.5 * (d == 0))[m].sum(); nd += m.sum()
            rows[-1].update(dict(discordant_pairs=int(nd), discordant_frac=nd / w.sum(), W_model_discordant=num_d / nd))
            gaprows.append(dict(dataset=ds, criterion=c, seed=s, gap_s=GAP_S, pairs_kept=int(npairs), pairs_all=int(w.sum()),
                                W_model_all=Wm, W_model_gap=num_m / npairs, W_time_all=Wt, W_time_gap=num_t / npairs))
R = pd.DataFrame(rows)
# two-sided p-values from the t approximation and Benjamini-Hochberg adjustment within each dataset (nine tests)
R["dW_se"] = (R.dW_thi - R.dW) / tdist.ppf(0.975, (R.neff - 1).clip(lower=1))
R["dW_tp"] = 2 * tdist.sf(np.abs(R.dW / R.dW_se), (R.neff - 1).clip(lower=1))
def bh(p):
    p = np.asarray(p); o = np.argsort(p); n = len(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)
R["dW_tq"] = R.groupby("dataset").dW_tp.transform(lambda s: bh(s.values))
R.to_csv(f"{OUT}/reference_anchored.csv", index=False)
G = pd.DataFrame(gaprows); G.to_csv(f"{OUT}/transition_distance.csv", index=False)
pd.set_option("display.width", 250)
print(R[["dataset", "criterion", "seed", "W_model", "W_time", "dW", "dW_tlo", "dW_thi", "dW_tp", "dW_tq", "discordant_frac", "W_model_discordant"]].round(3).to_string())
print(G.round(3).to_string())
