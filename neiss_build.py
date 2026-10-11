"""
Paper 4, step 2: label each injury by device, using product codes AND the injury story (narrative).

Why the stories matter: CPSC changed its product codes twice in this period.
  - 2019: powered scooters and hoverboards shared older codes (5042, 1329).
  - Until 2023: e-bikes were mixed with mopeds under code 3215 ("mopeds or power-assisted cycles").
  - From 2024: e-bikes got their own code (5045) and mopeds a new one (5046).
So the codes alone cannot give a consistent e-bike trend. The narrative text can.

Step A: rules (product codes + keywords in the story) give a device label.
Step B: a text classifier (TF-IDF + logistic regression) trained on 2024-2025 records,
        where codes 5045 (e-bike) and 5046 (moped) are separate, labels the 2019-2023 code-3215
        records that the rules could not decide.

Needs:  data/neiss_micromobility_2019_2025.csv  (from neiss_extract.py)
Output: neiss_labeled.csv
Run:    python neiss_build.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

BASE = Path(__file__).resolve().parent
d = pd.read_csv(BASE / "data" / "neiss_micromobility_2019_2025.csv", dtype=str)
d["narr"] = d["Narrative_1"].fillna("").str.upper()
d["year"] = d["file_year"].astype(int)
d["weight"] = pd.to_numeric(d["Weight"], errors="coerce")
codes = d[["Product_1", "Product_2", "Product_3"]].fillna("0")


def has_code(*cs):
    return codes.isin(list(cs)).any(axis=1)


def kw(pattern):
    pattern = pattern.replace("(", "(?:").replace("(?:?:", "(?:")
    return d["narr"].str.contains(pattern, regex=True)


E_WORDS = r"\bE-?BIKES?\b|\bE BIKES?\b|\bE-?BICYCLES?\b|ELECTRI(C|CAL) (BI(KE|CYCLE)|PEDAL|ASSIST)|PEDAL[- ]ASSIST|POWER[- ]ASSIST(ED)? (BI(KE|CYCLE))|BATTERY[- ]POWERED (BI(KE|CYCLE))"
ESC_WORDS = (r"\bE-?SCOOTERS?\b|\bE SCOOTERS?\b|ELEC(TRIC|TRONIC)? SCOOTER|MOTORI[ZS]ED SCOOTER|POWERED SCOOTER|"
             r"STAND[- ]?UP (ELECTRIC|MOTORIZED)|LIME SCOOTER|BIRD SCOOTER|RENTAL SCOOTER|BATTERY[- ]POWERED SCOOTER")
MOBILITY_WORDS = r"MOBILITY SCOOTER|INVALID|ASSIST(IVE|ED) MOBILITY|POWER CHAIR|WHEELCHAIR|HANDICAP"
MOPED_WORDS = r"MOPED|MINI ?BIKE|DIRT ?BIKE|MOTOR ?BIKE|MOTORCYCLE|GAS (POWERED|SCOOTER)|VESPA"
EXERCISE_WORDS = r"STATIONARY|EXERCISE BIKE|SPIN BIKE|PELOTON|WORKOUT BIKE"

label = pd.Series("Exclude", index=d.index)
bike_codes = has_code("5040", "5033")
scooter_codes = has_code("5022", "5023", "5024", "1329", "5042")
# Order matters: later rules override earlier ones.
label[bike_codes] = "Bicycle"
label[has_code("5024", "5023", "1329")] = "Kick scooter"
label[has_code("5022") | (has_code("5042") & kw(ESC_WORDS))] = "E-scooter"
label[has_code("5025") | (has_code("5042") & kw(r"HOVER ?BOARD"))] = "Hoverboard"
label[has_code("1333")] = "Skateboard"
label[has_code("5046", "5035") | (has_code("3215") & kw(MOPED_WORDS))] = "Moped / minibike"
label[has_code("5045")] = "E-bike"
# Story overrides inside bike and scooter codes
label[(bike_codes | has_code("3215")) & kw(E_WORDS)] = "E-bike"
label[scooter_codes & kw(ESC_WORDS)] = "E-scooter"
label[scooter_codes & kw(MOBILITY_WORDS)] = "Exclude"
label[kw(EXERCISE_WORDS) & label.isin(["Bicycle", "E-bike"])] = "Exclude"
label[bike_codes & ~has_code("5045", "3215") & kw(r"NON-? ?ELECTRIC")] = "Bicycle"
undecided = has_code("3215") & ~kw(E_WORDS) & ~kw(MOPED_WORDS)
label[undecided] = "Undecided 3215"
d["device"] = label
d["label_source"] = np.where(undecided, "classifier", "rules")

# ---- Step B: text classifier for the undecided 3215 records ----
train = d[has_code("5045", "5046") & (d["year"] >= 2024)].copy()
train["y"] = has_code("5045")[train.index].astype(int)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
Xtr = vec.fit_transform(train["narr"])
clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
cv_auc = cross_val_score(clf, Xtr, train["y"], cv=5, scoring="roc_auc").mean()
clf.fit(Xtr, train["y"])
if undecided.any():
    p = clf.predict_proba(vec.transform(d.loc[undecided, "narr"]))[:, 1]
    d.loc[undecided, "p_ebike"] = p
    d.loc[undecided, "device"] = np.where(p >= 0.5, "E-bike", "Moped / minibike")

# ---- Outcomes and circumstances from coded fields and the story ----
d["age"] = pd.to_numeric(d["Age"], errors="coerce").astype(float)
d.loc[d["age"] > 200, "age"] = (d.loc[d["age"] > 200, "age"] - 200) / 12  # NEISS codes infants' months as 2xx
d["hospitalized"] = d["Disposition"].isin(["2", "4", "5", "8"]).astype(int)  # transferred, admitted, held, died
d["died"] = (d["Disposition"] == "8").astype(int)
d["head_injury"] = d[["Body_Part", "Body_Part_2"]].isin(["75", "94", "76", "77"]).any(axis=1).astype(int)
d["helmet_no"] = kw(r"NO HELMET|UNHELMET|W/?O HELMET|WITHOUT (A )?HELMET|NOT WEARING (A )?HELMET|HELMET:? ?NO").astype(int)
d["helmet_yes"] = (kw(r"HELMET") & ~d["helmet_no"].astype(bool)).astype(int)
d["motor_vehicle"] = kw(r"\bMVC\b|\bMVA\b|STRUCK BY (A )?(CAR|VEHICLE|TRUCK|SUV)|HIT BY (A )?(CAR|VEHICLE|TRUCK|SUV)|"
                        r"\bVS\.? (CAR|VEHICLE|AUTO|TRUCK|SUV)|COLLIDED WITH (A )?(CAR|VEHICLE)").astype(int)
d["alcohol"] = (d["Alcohol"] == "1").astype(int)

d.drop(columns=["narr"]).to_csv(BASE / "neiss_labeled.csv", index=False)

pd.set_option("display.width", 200)
print(f"Text classifier (e-bike vs moped, trained on 2024-25 codes): cross-validated AUC = {cv_auc:.3f}")
print(f"Undecided code-3215 records labeled by the classifier: {int(undecided.sum()):,}\n")
keep = d[d["device"] != "Exclude"]
print("ER visits in the sample, by device and year")
print(pd.crosstab(keep["device"], keep["year"]).to_string())
print("\nNational estimates (sum of weights, thousands), by device and year")
print((keep.pivot_table(index="device", columns="year", values="weight", aggfunc="sum") / 1000).round(1).to_string())
print("\nSaved neiss_labeled.csv")
