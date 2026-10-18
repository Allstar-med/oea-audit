"""Endoscapes, original checkpoints: single-annotator reference standards, leave-one-operation-out W - E1,
and reference values for the ranking-disagreement fraction (seeds 17 and 29).
Usage: python endoscapes_sensitivity.py DATA_DIR OUT_DIR"""
import sys, os, ast
import numpy as np, pandas as pd
from scipy.stats import rankdata

D, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
L = pd.read_csv(f"{D}/endoscapes/endoscapes_test_linkage.csv")
R = pd.read_csv(f"{D}/endoscapes/endoscapes_test_annotator_ratings.csv")
K = L.merge(R, on=["video", "frame"], how="left", validate="one_to_one")
assert K.cvs_annotator_1.notna().all()


def auc(p, n):
    r = rankdata(np.r_[p, n])
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def estimands(df, y, s):
    e1 = auc(df.loc[df[y] == 1, s].values, df.loc[df[y] == 0, s].values)
    a, w = [], []
    for _, g in df.groupby("video"):
        p, n = g.loc[g[y] == 1, s].values, g.loc[g[y] == 0, s].values
        if len(p) and len(n):
            a.append(auc(p, n)); w.append(len(p) * len(n))
    a, w = np.array(a), np.array(w)
    return dict(E1=e1, W=(a * w).sum() / w.sum(), E2=a.mean(), mixed=len(a), positives=int(df[y].sum()))


rows = []
for k in (1, 2, 3):
    r = K[f"cvs_annotator_{k}"].apply(ast.literal_eval)
    for c in (1, 2, 3):
        K[f"a{k}_c{c}"] = r.apply(lambda v: int(v[c - 1] >= 0.5))
for c in (1, 2, 3):
    for s in (17, 29, 43):
        for ref in ("annotator 1", "annotator 2", "annotator 3", "majority"):
            y = f"y_c{c}" if ref == "majority" else f"a{ref[-1]}_c{c}"
            e = estimands(K, y, f"p_c{c}_seed{s}")
            rows.append(dict(criterion=f"C{c}", seed=s, reference=ref, **e, W_minus_E1=e["W"] - e["E1"], E2_minus_E1=e["E2"] - e["E1"]))
pd.DataFrame(rows).to_csv(f"{OUT}/endoscapes_annotator_sensitivity.csv", index=False)

rows = []
for c in (1, 2, 3):
    for s in (17, 29, 43):
        y, sc = f"y_c{c}", f"p_c{c}_seed{s}"
        full = estimands(L, y, sc); d0 = full["W"] - full["E1"]
        loo = {v: (lambda e: e["W"] - e["E1"])(estimands(L[L.video != v], y, sc)) for v in sorted(L.video.unique())}
        vals = np.array(list(loo.values()))
        rows.append(dict(criterion=f"C{c}", seed=s, W_minus_E1=d0, loo_min=vals.min(), loo_max=vals.max(),
                         op_min=min(loo, key=loo.get), op_max=max(loo, key=loo.get), sign_changes=int((np.sign(vals) != np.sign(d0)).sum())))
pd.DataFrame(rows).to_csv(f"{OUT}/endoscapes_leave_one_operation_out.csv", index=False)

rows = []
for c in (1, 2, 3):
    tot = opp = ea = eb = ind = 0.0
    for _, g in L.groupby("video"):
        p, n = g[g[f"y_c{c}"] == 1], g[g[f"y_c{c}"] == 0]
        if len(p) == 0 or len(n) == 0:
            continue
        da = p[f"p_c{c}_seed17"].values[:, None] - n[f"p_c{c}_seed17"].values[None, :]
        db = p[f"p_c{c}_seed29"].values[:, None] - n[f"p_c{c}_seed29"].values[None, :]
        opp += (np.sign(da) != np.sign(db)).sum(); tot += da.size
        xa, xb = (da < 0).mean(), (db < 0).mean()
        ea += xa * da.size; eb += xb * da.size; ind += da.size * (xa * (1 - xb) + xb * (1 - xa))
    rows.append(dict(criterion=f"C{c}", pairs=int(tot), opposite=opp / tot, error_seed17=ea / tot, error_seed29=eb / tot,
                     lower_bound=abs(ea - eb) / tot, upper_bound=(ea + eb) / tot, independent_errors=ind / tot))
pd.DataFrame(rows).to_csv(f"{OUT}/endoscapes_disagreement_reference.csv", index=False)
print(pd.read_csv(f"{OUT}/endoscapes_disagreement_reference.csv").round(3).to_string())
print(pd.read_csv(f"{OUT}/endoscapes_leave_one_operation_out.csv").round(3).to_string())
