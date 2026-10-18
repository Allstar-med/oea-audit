"""SAGES C2 model comparisons rerun with three further random streams.
Usage: python pvalue_streams.py DATA_DIR OUT_CSV"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import oea_audit as o

SRC, OUT = sys.argv[1], sys.argv[2]
L = pd.read_csv(f"{SRC}/sages/sages_labels.csv"); L = L[L.primary_include == 1]
Pr = pd.read_csv(f"{SRC}/sages/sages_predictions.csv")
d = L.merge(Pr, on=["dataset_version", "split", "operation_or_video_id", "clip_id", "frame_id"])
df = d.assign(model=d.model_id.str.extract(r"(seed\d+)")[0], y=d["c2_majority"].astype(int), score=d["p_c2"])
out = []
for seed in (1, 2, 3):
    est, cmp = o.run(df, "clip_id", "y", "score", "model", 10000, 424242 + seed)
    cmp["rng"] = seed; out.append(cmp)
    print(cmp[["estimand", "model_a", "model_b", "p_two_sided"]].to_string(), flush=True)
pd.concat(out).to_csv(OUT, index=False)
