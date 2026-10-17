"""Builds the manuscript and supplementary tables (pandoc pipe tables) from the frozen CSVs."""
import sys
import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
A = pd.read_csv(f"{ROOT}/Tables/All_18_estimates.csv")
B = pd.read_csv(f"{ROOT}/Tables/SAGES_10000_bootstrap.csv")
B["criterion"] = B.criterion.str.upper(); B["seed"] = B.model.str.extract(r"seed(\d+)").astype(int)
M = pd.read_csv(f"{ROOT}/Tables/27_model_comparisons.csv")
M["criterion"] = M.criterion.str.upper()
M["a"] = M.model_a.str.extract(r"seed(\d+)").astype(int); M["b"] = M.model_b.str.extract(r"seed(\d+)").astype(int)
R = pd.read_csv(f"{ROOT}/Tables/Rater_sensitivity.csv")
R["criterion"] = R.criterion.str.upper(); R["seed"] = R.model.str.extract(r"seed(\d+)").astype(int)
P = pd.read_csv(f"{ROOT}/Tables/Pair_disagreement.csv"); P["criterion"] = P.criterion.str.upper()

from decimal import Decimal, ROUND_HALF_UP
def h(v, n=3):
    return str(Decimal(repr(float(v))).quantize(Decimal(10) ** -n, rounding=ROUND_HALF_UP))
f3 = lambda v: h(v)
sg = lambda v: "0.000" if f"{abs(v):.3f}" == "0.000" else ("+" if v >= 0 else "−") + f"{abs(v):.3f}"
ci = lambda lo, hi: f"[{sg(lo)}, {sg(hi)}]"
ci_plain = lambda lo, hi: f"[{h(lo)}, {h(hi)}]"


def pipe(df, align=None):
    cols = list(df.columns)
    al = align or ["l"] * len(cols)
    sep = ["---" if a == "l" else "--:" if a == "r" else ":-:" for a in al]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(sep) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    return "\n".join(lines)


T = {}
T["T1"] = pipe(pd.DataFrame([
    ["$E_1$", "All $N^+N^-$ positive–negative pairs", "None; frames pooled across units", "Separation of positive and negative frames across the pooled test set"],
    ["$X$", "Pairs whose frames come from different units", "None; frames pooled across units", "Separation across recordings only"],
    ["$W$", "Pairs within the same unit", "$w_u=n_u^+n_u^-$", "Within-recording discrimination, large mixed units weighted more"],
    ["$E_2$", "Pairs within the same unit", "$1/m$ for each mixed-label unit", "Within-recording discrimination of a typical mixed-label unit"],
], columns=["Estimand", "Pairs compared", "Unit weight", "Question answered"]))

T["T2"] = pipe(pd.DataFrame([
    ["1", "Clinical comparison and matching estimand", "Within-operation ranking of CVS states; $W$ and $E_2$ alongside $E_1$", "Same"],
    ["2", "Observable clustering unit; link to patient", "Operation (one video per operation)", "90-s clip, one per donated video; patient and surgeon links unavailable"],
    ["3", "Within-unit pair fraction $\\alpha$", "2.75–2.97%", "0.17–0.19%"],
    ["4", "Mixed-label units per criterion", "24, 19, 22 of 40", "92, 158, 76 of 293"],
    ["5", "Weight concentration of $W$ ($n_{\\mathrm{eff}}$)", "4.7, 5.8, 6.2", "79.9, 140.0, 67.5"],
    ["6", "$E_1$, $X$, $W$, $E_2$ with unit-level intervals", "10,000 matched operation resamples; $t$ interval for $W$", "10,000 matched clip resamples; country-level sensitivity"],
    ["7", "Contrasts $W-E_1$, $E_2-E_1$, $E_2-W$", "Section 5.3", "Section 5.3"],
    ["8", "Paired model comparisons, multiplicity control", "27 comparisons, Benjamini–Hochberg within family", "27 comparisons, Benjamini–Hochberg within family"],
    ["9", "Reference-standard sensitivity", "Each of three annotators; leave-one-operation-out", "Each of three rating columns"],
    ["10", "Numerical check of $E_1=\\alpha W+(1-\\alpha)X$", "Max. error $<10^{-15}$", "Max. error $<10^{-15}$"],
    ["11", "$\\Delta W$ against a time baseline", "Section 5.8", "Section 5.8"],
], columns=["Item", "Report", "Endoscapes", "SAGES 2024"]))

el = A.groupby(["dataset", "criterion"]).eligible.first()
T["T3"] = pipe(pd.DataFrame([
    ["Recording unit", "Operation (one video)", "Clip (90 s; one per donated video)"],
    ["Licence", "CC BY-NC-SA 4.0", "CC BY-NC 4.0"],
    ["Units in analysis", "40", "293 (300 before duplicate exclusion)"],
    ["Annotated frames", "1,663 of 1,799 keyframes", "5,274 (18 per clip)"],
    ["Annotators per frame", "3", "3"],
    ["Primary reference label", "Majority vote", "Majority vote"],
    ["Positive frames C1 / C2 / C3", "418 / 294 / 497", "570 / 1,244 / 545"],
    ["Mixed-label units C1 / C2 / C3", " / ".join(str(int(el[("Endoscapes", c)])) for c in ["C1", "C2", "C3"]),
     " / ".join(str(int(el[("SAGES 2024", c)])) for c in ["C1", "C2", "C3"])],
    ["Models (seeds 17, 29, 43)", "Original ResNet-18 checkpoints", "ResNet-18 retrained on Endoscapes only, no adaptation to SAGES"],
], columns=["Characteristic", "Endoscapes (test split)", "SAGES 2024 (test set)"]))

rows = []
for _, r in A.iterrows():
    rows.append([r.dataset.replace(" 2024", ""), r.criterion, int(r.seed), f3(r.E1), f3(r.X), f3(r.W), f3(r.E2),
                 h(r.alpha*100, 2), int(r.eligible), sg(r.W_minus_E1), sg(r.E2_minus_E1)])
T["T4"] = pipe(pd.DataFrame(rows, columns=["Dataset", "Criterion", "Seed", "$E_1$", "$X$", "$W$", "$E_2$", "$\\alpha$ (%)",
                                           "Mixed units", "$W-E_1$", "$E_2-E_1$"]), ["l", "l", "r"] + ["r"] * 8)

rows = []
for _, r in B.iterrows():
    rows.append([r.criterion, int(r.seed), f"{h(r.E1)} {ci_plain(r.E1_lo, r.E1_hi)}", f"{h(r.W)} {ci_plain(r.W_lo, r.W_hi)}",
                 f"{h(r.E2)} {ci_plain(r.E2_lo, r.E2_hi)}", f"{sg(r.W_minus_E1)} {ci(r.W_minus_E1_lo, r.W_minus_E1_hi)}",
                 f"{sg(r.E2_minus_E1)} {ci(r.E2_minus_E1_lo, r.E2_minus_E1_hi)}"])
T["T5"] = pipe(pd.DataFrame(rows, columns=["Criterion", "Seed", "$E_1$ [95% CI]", "$W$ [95% CI]", "$E_2$ [95% CI]",
                                           "$W-E_1$ [95% CI]", "$E_2-E_1$ [95% CI]"]), ["l", "r"] + ["r"] * 5)

# Supplementary
rows = []
for est in ["E1", "W", "E2"]:
    for c in ["C1", "C2", "C3"]:
        for (x, y) in [(17, 29), (17, 43), (29, 43)]:
            r = M[(M.estimand == est) & (M.criterion == c) & (M.a == x) & (M.b == y)].iloc[0]
            rows.append([f"${est[0]}_{est[1]}$" if est != "W" else "$W$", c, f"{x} vs {y}", sg(r.difference), ci(r.ci_lo, r.ci_hi),
                         h(r.p_two_sided, 4), h(r.q_BH_within_estimand, 4)])
T["S1"] = pipe(pd.DataFrame(rows, columns=["Estimand", "Criterion", "Seeds", "Difference", "95% CI", "$p$", "$q$ (BH)"]),
               ["l", "l", "l", "r", "r", "r", "r"])
rows = []
for _, r in R.sort_values(["criterion", "seed", "rater"]).iterrows():
    rows.append([r.criterion, int(r.seed), int(r.rater), f3(r.E1), f3(r.W), f3(r.E2), h(r.alpha*100, 2), int(r.estimable),
                 sg(r.W_minus_E1), sg(r.E2_minus_E1)])
T["S2"] = pipe(pd.DataFrame(rows, columns=["Criterion", "Seed", "Column", "$E_1$", "$W$", "$E_2$", "$\\alpha$ (%)", "Mixed clips",
                                           "$W-E_1$", "$E_2-E_1$"]), ["l"] + ["r"] * 9)
rows = []
for _, r in P.sort_values(["criterion", "video"]).iterrows():
    rows.append([r.criterion, int(r.video), int(r.positive_frames), int(r.negative_frames), f"{int(r.within_pairs):,}",
                 f"{int(r.opposite_order_pairs):,}", h(r.opposite_fraction*100, 1)])
T["S3"] = pipe(pd.DataFrame(rows, columns=["Criterion", "Operation", "Positive frames", "Negative frames", "Within pairs",
                                           "Oppositely ordered", "Fraction (%)"]), ["l"] + ["r"] * 6)
for k, v in T.items():
    open(f"{OUT}/{k}.md", "w").write(v + "\n")
print("tables:", list(T))
