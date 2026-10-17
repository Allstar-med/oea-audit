"""Sensitivity: SAGES W-E1 with countries (not clips) as the resampling unit; estimands still defined per clip.
python country_bootstrap.py <data_dir> <out_csv>"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
SRC, OUT = sys.argv[1], sys.argv[2]
L = pd.read_csv(f"{SRC}/sages/sages_labels.csv"); L = L[L.primary_include == 1]
Pr = pd.read_csv(f"{SRC}/sages/sages_predictions.csv")
V = pd.read_csv(f"{SRC}/sages/video_metadata.csv")[["video_id", "country"]]
d = L.merge(Pr, on=["dataset_version", "split", "operation_or_video_id", "clip_id", "frame_id"]).merge(V, left_on="clip_id", right_on="video_id")
assert len(d) == 5274 * 3


def wauc(y, s, w):
    pos = w * y; neg = w * (1 - y); P = pos.sum(); Q = neg.sum()
    o = np.argsort(s, kind="stable"); s = s[o]; pos = pos[o]; neg = neg[o]
    st = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    pg = np.add.reduceat(pos, st); ng = np.add.reduceat(neg, st)
    return ((np.cumsum(ng) - ng + 0.5 * ng) * pg).sum() / (P * Q)


rng = np.random.default_rng(99001); B = 10000
rows = []
for c in ("c1", "c2", "c3"):
    for m, g in d.groupby("model_id"):
        g = g.sort_values(["clip_id", "frame_id"])
        clips = g.clip_id.to_numpy(); uc, ci = np.unique(clips, return_inverse=True)
        ctry = g.groupby("clip_id").country.first().reindex(uc).to_numpy()
        uct, cti = np.unique(ctry, return_inverse=True)
        y = g[f"{c}_majority"].to_numpy().astype(float); s = g[f"p_{c}"].to_numpy()
        npos = np.bincount(ci, y); nneg = np.bincount(ci, 1 - y)
        A = np.full(len(uc), np.nan)
        for k in range(len(uc)):
            idx = ci == k
            if npos[k] > 0 and nneg[k] > 0:
                A[k] = wauc(y[idx], s[idx], np.ones(idx.sum()))
        w = npos * nneg
        def est(cw):
            fw = cw[ci]; e1 = wauc(y, s, fw); ww = cw * w; W = np.nansum(ww * A) / ww.sum(); return W - e1
        pt = est(np.ones(len(uc)))
        rng_local = np.random.default_rng(99001)
        draws = []
        for _ in range(B):
            cnt = np.bincount(rng_local.integers(0, len(uct), len(uct)), minlength=len(uct))
            cw = cnt[cti].astype(float)
            if (cw * w).sum() == 0: continue
            draws.append(est(cw))
        lo, hi = np.quantile(draws, [0.025, 0.975])
        rows.append(dict(criterion=c.upper(), model=m, countries=len(uct), W_minus_E1=pt, lo=lo, hi=hi, valid=len(draws)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
