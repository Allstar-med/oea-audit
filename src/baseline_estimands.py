"""E1, X, W and E2 of a score equal to the elapsed time of each frame within its recording.
Usage: python baseline_estimands.py DATA_DIR OUT_CSV"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

D, OUT = sys.argv[1:3]


def auc(p, n):
    r = rankdata(np.r_[p, n])
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def estimands(df, y, s, unit):
    pos, neg = df[df[y] == 1], df[df[y] == 0]
    e1 = auc(pos[s].values, neg[s].values)
    a, w = [], []
    for _, g in df.groupby(unit):
        p, n = g.loc[g[y] == 1, s].values, g.loc[g[y] == 0, s].values
        if len(p) and len(n):
            a.append(auc(p, n)); w.append(len(p) * len(n))
    a, w = np.array(a), np.array(w)
    npn = len(pos) * len(neg); alpha = w.sum() / npn
    W = (a * w).sum() / w.sum()
    X = (e1 - alpha * W) / (1 - alpha)
    return dict(E1=e1, X=X, W=W, E2=a.mean(), mixed=len(a))


rows = []
L = pd.read_csv(f"{D}/endoscapes/endoscapes_test_linkage.csv")
L["t"] = L.frame - L.groupby("video").frame.transform("min")  # time from the first annotated frame (frame units)
for c in (1, 2, 3):
    rows.append(dict(dataset="Endoscapes", criterion=f"C{c}", **estimands(L, f"y_c{c}", "t", "video")))
S = pd.read_csv(f"{D}/sages/sages_labels.csv"); S = S[S.primary_include == 1]
for c in (1, 2, 3):
    rows.append(dict(dataset="SAGES 2024", criterion=f"C{c}", **estimands(S, f"c{c}_majority", "timestamp", "clip_id")))
R = pd.DataFrame(rows); R.to_csv(OUT, index=False); print(R.round(3).to_string())
