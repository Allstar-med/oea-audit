"""Independent re-analysis of the row-level files (second implementation, own RNG streams).
python independent_v2.py <data_dir> <out_dir>"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import oea_audit as o
SRC, OUT = sys.argv[1], sys.argv[2]
B = 10000
rows = []; comps = []
# ---------------- Endoscapes (historical checkpoints)
D = {s: pd.read_csv(f"{SRC}/endoscapes/locked_test_predictions_seed{s}.csv").sort_values(["video", "frame"]).reset_index(drop=True) for s in (17, 29, 43)}
for s in (29, 43):
    assert D[17][["video", "frame"]].equals(D[s][["video", "frame"]])
    assert np.array_equal(D[17][["y_c1", "y_c2", "y_c3"]].values, D[s][["y_c1", "y_c2", "y_c3"]].values)
info = {"endoscapes_rows": len(D[17]), "endoscapes_ops": int(D[17].video.nunique()),
        "endoscapes_positives": {c: int(D[17][f"y_{c}"].sum()) for c in ("c1", "c2", "c3")}}
for k, c in enumerate(("c1", "c2", "c3")):
    df = pd.concat([D[s].assign(model=f"seed{s}", y=D[s][f"y_{c}"].astype(int), score=D[s][f"p_{c}"]) for s in (17, 29, 43)])
    est, cmp = o.run(df, "video", "y", "score", "model", B, 7700 + k)
    rows.append(est.assign(dataset="Endoscapes", criterion=c.upper())); comps.append(cmp.assign(dataset="Endoscapes", criterion=c.upper()))
    print("endo", c, flush=True)
# ---------------- SAGES (retrained source-only checkpoints)
L = pd.read_csv(f"{SRC}/sages/sages_labels.csv"); Pr = pd.read_csv(f"{SRC}/sages/sages_predictions.csv")
L = L[L.primary_include == 1]
d = L.merge(Pr, on=["dataset_version", "split", "operation_or_video_id", "clip_id", "frame_id"], validate="one_to_many")
assert len(d) == 5274 * 3
info.update(sages_rows=int(len(L)), sages_clips=int(L.clip_id.nunique()),
            sages_positives={c: int(L[f"{c}_majority"].sum()) for c in ("c1", "c2", "c3")})
for k, c in enumerate(("c1", "c2", "c3")):
    df = d.assign(model=d.model_id.str.extract(r"(seed\d+)")[0], y=d[f"{c}_majority"].astype(int), score=d[f"p_{c}"])
    est, cmp = o.run(df, "clip_id", "y", "score", "model", B, 8800 + k)
    rows.append(est.assign(dataset="SAGES 2024", criterion=c.upper())); comps.append(cmp.assign(dataset="SAGES 2024", criterion=c.upper()))
    print("sages", c, flush=True)
R = pd.concat(rows); Cm = pd.concat(comps)
# BH within dataset x estimand family (9 comparisons each)
Cm["q_BH"] = Cm.groupby(["dataset", "estimand"]).p_two_sided.transform(o.bh)
R.to_csv(f"{OUT}/independent_estimands_ci.csv", index=False); Cm.to_csv(f"{OUT}/independent_model_comparisons.csv", index=False)
json.dump(info, open(f"{OUT}/independent_info.json", "w"), indent=1)
print(json.dumps(info))
