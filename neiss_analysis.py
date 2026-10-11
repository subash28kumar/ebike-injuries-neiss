"""
Paper 4, step 3: national estimates, severity and circumstances, with survey-correct 95% CIs.

NEISS is a stratified sample of hospitals (Stratum, PSU = hospital, Weight). Standard errors use
Taylor linearization with hospitals as clusters within strata (the method CPSC recommends).

  1. Yearly national ER visits by device, 2019-2025 (with 95% CI)
  2. Severity by device: hospitalized, head injury (weighted %, 95% CI)
  3. Circumstances from the injury story: motor vehicle involved, no helmet / helmet mentioned, alcohol
  4. Odds of hospitalization for e-bike and e-scooter vs regular bicycle, adjusted for age group, sex,
     year, motor vehicle and alcohol (weighted logistic regression, SEs clustered by hospital)

Needs: neiss_labeled.csv (from neiss_build.py)
Run:   python neiss_analysis.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
MAIN = ["E-bike", "E-scooter", "Bicycle", "Kick scooter"]

d = pd.read_csv(BASE / "neiss_labeled.csv", low_memory=False)
d = d[d["device"] != "Exclude"].copy()
d["w"] = d["weight"]
d["agegrp"] = pd.cut(d["age"], [-1, 17.99, 64.99, 200], labels=["0-17", "18-64", "65+"])
d["male"] = (d["Sex"] == 1).astype(int)


def total_ci(mask, data=d):
    """Weighted total of records in mask, with linearized SE (hospitals within strata)."""
    y = data["w"] * mask.astype(float)
    g = pd.DataFrame({"s": data["Stratum"], "p": data["PSU"], "y": y}).groupby(["s", "p"])["y"].sum().reset_index()
    var = 0.0
    for _, h in g.groupby("s"):
        n = len(h)
        if n > 1:
            var += n / (n - 1) * ((h["y"] - h["y"].mean()) ** 2).sum()
    t = y.sum()
    se = np.sqrt(var)
    return t, t - 1.96 * se, t + 1.96 * se


def prop_ci(num_mask, den_mask, data=d):
    """Weighted proportion num/den with linearized SE (ratio estimator)."""
    x = data["w"] * den_mask.astype(float)
    y = data["w"] * (num_mask & den_mask).astype(float)
    R = y.sum() / x.sum()
    z = (y - R * x) / x.sum()
    g = pd.DataFrame({"s": data["Stratum"], "p": data["PSU"], "z": z}).groupby(["s", "p"])["z"].sum().reset_index()
    var = 0.0
    for _, h in g.groupby("s"):
        n = len(h)
        if n > 1:
            var += n / (n - 1) * ((h["z"] - h["z"].mean()) ** 2).sum()
    se = np.sqrt(var)
    return R, max(R - 1.96 * se, 0), min(R + 1.96 * se, 1)


# ---- 1. Yearly national estimates ----
rows = []
for dev in d["device"].unique():
    for yr in range(2019, 2026):
        t, lo, hi = total_ci((d["device"] == dev) & (d["year"] == yr))
        rows.append({"device": dev, "year": yr, "estimate": t, "ci_low": lo, "ci_high": hi,
                     "sample_n": int(((d["device"] == dev) & (d["year"] == yr)).sum())})
trend = pd.DataFrame(rows)
trend.to_csv(BASE / "trend_estimates.csv", index=False)

# ---- 2-3. Severity and circumstances by device (2019-2025 pooled) ----
measures = {"Hospitalized": "hospitalized", "Head injury": "head_injury", "Motor vehicle involved": "motor_vehicle",
            "No helmet (stated)": "helmet_no", "Helmet worn (stated)": "helmet_yes", "Alcohol involved": "alcohol"}
srows = []
for dev in MAIN:
    den = d["device"] == dev
    r = {"device": dev, "sample_n": int(den.sum()),
         "median_age": float(np.median(np.repeat(d.loc[den, "age"].values, 1)))}
    for lab, col in measures.items():
        p, lo, hi = prop_ci(d[col] == 1, den)
        r[lab] = f"{p*100:.1f} ({lo*100:.1f}-{hi*100:.1f})"
        r[lab + "_p"] = p
    srows.append(r)
sev = pd.DataFrame(srows)
sev.to_csv(BASE / "severity_by_device.csv", index=False)

# Hospitalized by age group
arows = []
for dev in MAIN:
    for ag in ["0-17", "18-64", "65+"]:
        den = (d["device"] == dev) & (d["agegrp"] == ag)
        if den.sum() < 30:
            continue
        p, lo, hi = prop_ci(d["hospitalized"] == 1, den)
        share, _, _ = prop_ci(d["agegrp"] == ag, d["device"] == dev)
        arows.append({"device": dev, "age": ag, "share_of_injuries": share, "hospitalized": p,
                      "lo": lo, "hi": hi, "sample_n": int(den.sum())})
age = pd.DataFrame(arows)
age.to_csv(BASE / "hospitalized_by_age.csv", index=False)

# ---- 4. Adjusted odds of hospitalization vs bicycle ----
m = d[d["device"].isin(MAIN) & d["agegrp"].notna()].copy()
m["device"] = pd.Categorical(m["device"], categories=["Bicycle", "E-bike", "E-scooter", "Kick scooter"])
m["wn"] = m["w"] / m["w"].mean()
fit = smf.glm("hospitalized ~ C(device) + C(agegrp) + male + C(year) + motor_vehicle + alcohol", data=m,
              family=sm.families.Binomial(), var_weights=m["wn"]).fit(
    cov_type="cluster", cov_kwds={"groups": m["Stratum"].astype(str) + "_" + m["PSU"].astype(str)})
orr = []
for v in fit.params.index:
    if v.startswith("C(device)") or v in ("motor_vehicle", "alcohol") or v.startswith("C(agegrp)"):
        b, se = fit.params[v], fit.bse[v]
        orr.append({"term": v.replace("C(device)[T.", "").replace("C(agegrp)[T.", "Age ").rstrip("]"),
                    "odds_ratio": np.exp(b), "ci_low": np.exp(b - 1.96 * se), "ci_high": np.exp(b + 1.96 * se),
                    "p": fit.pvalues[v]})
orr = pd.DataFrame(orr)
orr.to_csv(BASE / "hospitalization_odds.csv", index=False)

# ---- Print ----
pd.set_option("display.width", 220)
pv = trend[trend["device"].isin(MAIN)].copy()
pv["txt"] = pv.apply(lambda r: f"{r['estimate']/1000:.1f} ({r['ci_low']/1000:.1f}-{r['ci_high']/1000:.1f})", axis=1)
print("1. National ER visits, thousands (95% CI)\n")
print(pv.pivot(index="device", columns="year", values="txt").to_string())
print("\n2-3. Severity and circumstances, 2019-2025 pooled, weighted % (95% CI)\n")
print(sev[["device", "sample_n", "median_age"] + list(measures)].to_string(index=False))
print("\nHospitalized by age group (weighted %)\n")
a = age.copy()
a["hosp"] = a.apply(lambda r: f"{r['hospitalized']*100:.1f} ({r['lo']*100:.1f}-{r['hi']*100:.1f})", axis=1)
a["share"] = (a["share_of_injuries"] * 100).round(1)
print(a.pivot(index="device", columns="age", values="hosp").to_string())
print("\nShare of injuries by age group (%)\n")
print(a.pivot(index="device", columns="age", values="share").to_string())
print("\n4. Adjusted odds of hospitalization (vs regular bicycle; ages vs 0-17)\n")
o = orr.copy()
o["OR (95% CI)"] = o.apply(lambda r: f"{r['odds_ratio']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f})", axis=1)
o["p"] = o["p"].round(4)
print(o[["term", "OR (95% CI)", "p"]].to_string(index=False))
print("\nSaved trend_estimates.csv, severity_by_device.csv, hospitalized_by_age.csv, hospitalization_odds.csv")
