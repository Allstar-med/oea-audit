"""Table 7 (reference-anchored within-unit AUROC). python make_t7.py <reference_anchored.csv> <table_dir>"""
import sys
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

R = pd.read_csv(sys.argv[1]); OUT = sys.argv[2]
h = lambda v, n=3: str(Decimal(repr(float(v))).quantize(Decimal(10) ** -n, rounding=ROUND_HALF_UP))
sg = lambda v: "0.000" if h(abs(v)) == "0.000" else ("+" if v >= 0 else "−") + h(abs(v))
rows = []
for _, r in R.iterrows():
    rows.append([r.dataset.replace(" 2024", ""), r.criterion, int(r.seed), f"{h(r.W_model)} [{h(r.W_model_tlo)}, {h(r.W_model_thi)}]",
                 h(r.W_time), f"{sg(r.dW)} [{sg(r.dW_tlo)}, {sg(r.dW_thi)}]", h(r.dW_tq), f"{sg(r.dE2)} [{sg(r.dE2_tlo)}, {sg(r.dE2_thi)}]",
                 h(r.W_model_discordant)])
cols = ["Dataset", "Crit.", "Seed", "$W$ model [$t$ interval]", "$W$ time", "$\\Delta W$ [$t$ interval]", "$q$", "$\\Delta E_2$ [$t$ interval]", "$W$ model, time-discordant pairs"]
al = ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]
lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(al) + " |"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
open(f"{OUT}/T7.md", "w").write("\n".join(lines) + "\n")
print("T7 written")
