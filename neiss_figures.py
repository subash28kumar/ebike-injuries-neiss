"""
Paper 4, step 4: journal-style figures (same style as Papers 1-3).

  Figure 1  National ER visits for e-bikes and e-scooters, 2019-2025 (95% CI)       (trend_estimates.csv)
  Figure 2  E-bike visits by age group, with children's visits labeled                (neiss_labeled.csv)
  Figure 3  Adjusted odds of hospitalization vs regular bicycle, by age group          (neiss_labeled.csv)
  Figure A1 E-bike visits by how they were found: product codes vs AI text classifier (neiss_labeled.csv)

Run after neiss_build.py and neiss_analysis.py:
    python neiss_figures.py
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.8, "legend.frameon": False, "savefig.dpi": 600})
EBIKE, ESCOOT, GREY = "#0072b2", "#d55e00", "#999999"


def clean(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def save(fig, name):
    kw = {"bbox_inches": "tight", "pad_inches": 0.05}
    fig.savefig(OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    fig.savefig(OUT / f"{name}.pdf", **kw)
    fig.savefig(OUT / f"{name}.png", dpi=200, **kw)
    plt.close(fig)
    print(f"Saved {name}")


# ---------- Figure 1 ----------
t = pd.read_csv(BASE / "trend_estimates.csv")
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for dev, col, mk in [("E-scooter", ESCOOT, "s"), ("E-bike", EBIKE, "o"), ("Kick scooter", GREY, "^")]:
    s = t[t["device"] == dev].sort_values("year")
    ax.fill_between(s["year"], s["ci_low"] / 1000, s["ci_high"] / 1000, color=col, alpha=0.15, linewidth=0)
    ax.plot(s["year"], s["estimate"] / 1000, color=col, marker=mk, markersize=3.5, linewidth=1.2, label=dev)
    last = s.iloc[-1]
    ax.text(2025.15, last["estimate"] / 1000, f"{last['estimate']/1000:.0f}k", color=col, va="center", fontsize=7)
ax.set_xticks(range(2019, 2026))
ax.set_xlim(2018.7, 2025.7)
ax.set_ylim(0, None)
ax.set_ylabel("Estimated U.S. ER visits (thousands)")
ax.legend(loc="upper left")
clean(ax)
fig.tight_layout()
save(fig, "Figure1")

# ---------- Figure A1 ----------
d = pd.read_csv(BASE / "neiss_labeled.csv", dtype={"Product_1": str, "Product_2": str, "Product_3": str}, low_memory=False)
c = d[["Product_1", "Product_2", "Product_3"]].fillna("0")
e = d[d["device"] == "E-bike"].copy()
ce = c.loc[e.index]
e["src"] = np.select(
    [ce.isin(["5045"]).any(axis=1), ce.isin(["3215"]).any(axis=1) & (e["label_source"] == "classifier"),
     ce.isin(["3215"]).any(axis=1)],
    ["New e-bike code 5045", "Old code 3215, sorted by the AI text classifier", "Old code 3215, sorted by keywords"],
    "Bicycle code, e-bike named in the story")
order = ["Old code 3215, sorted by keywords", "Old code 3215, sorted by the AI text classifier", "New e-bike code 5045"]
p = e.pivot_table(index="year", columns="src", values="weight", aggfunc="sum").reindex(columns=order).fillna(0) / 1000
fig, ax = plt.subplots(figsize=(3.5, 2.6))
bottom = np.zeros(len(p))
for col, alpha in zip(order, [0.55, 0.25, 1.0]):
    ax.bar(p.index, p[col], bottom=bottom, color=EBIKE, alpha=alpha, width=0.65, label=col)
    bottom += p[col].values
ax.axvline(2023.5, color=GREY, linewidth=0.8, linestyle="--")
ax.set_xticks(range(2019, 2026))
ax.set_ylabel("Estimated e-bike ER visits (thousands)")
ax.legend(loc="upper left", fontsize=6.5)
clean(ax)
fig.tight_layout()
save(fig, "FigureA1")

# ---------- Figure 2: e-bike injuries by age group ----------
e2 = d[d["device"] == "E-bike"].copy()
e2["agegrp"] = pd.cut(e2["age"], [-1, 17.99, 64.99, 200], labels=["Children (0-17)", "Adults (18-64)", "Older adults (65+)"])
q = e2.pivot_table(index="year", columns="agegrp", values="weight", aggfunc="sum").fillna(0) / 1000
fig, ax = plt.subplots(figsize=(3.5, 2.6))
bottom = np.zeros(len(q))
for col, alpha in zip(["Children (0-17)", "Adults (18-64)", "Older adults (65+)"], [1.0, 0.45, 0.2]):
    ax.bar(q.index, q[col], bottom=bottom, color=EBIKE, alpha=alpha, width=0.65, label=col)
    bottom += q[col].values
for yr in q.index:
    kids = q.loc[yr, "Children (0-17)"]
    ax.text(yr, bottom[list(q.index).index(yr)] + 1.5, f"{kids:.1f}k", ha="center", fontsize=6.5, color=EBIKE)
ax.set_xticks(range(2019, 2026))
ax.set_ylabel("Estimated e-bike ER visits (thousands)")
ax.text(2018.6, bottom.max() * 1.02, "Numbers above bars: children", fontsize=6.5, color=EBIKE, va="bottom")
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=6.5)
clean(ax)
fig.tight_layout()
save(fig, "Figure2")

# ---------- Figure 3 ----------
m = d[d["device"].isin(["Bicycle", "E-bike", "E-scooter", "Kick scooter"])].copy()
m["agegrp"] = pd.cut(m["age"], [-1, 17.99, 64.99, 200], labels=["0-17", "18-64", "65+"])
m = m[m["agegrp"].notna()]
m["male"] = (m["Sex"] == 1).astype(int)
m["wn"] = m["weight"] / m["weight"].mean()
m["cl"] = m["Stratum"].astype(str) + "_" + m["PSU"].astype(str)
m["device"] = pd.Categorical(m["device"], ["Bicycle", "E-bike", "E-scooter", "Kick scooter"])
rows = []
for ag, lab in [("0-17", "Children (0-17)"), ("18-64", "Adults (18-64)"), ("65+", "Older adults (65+)")]:
    s = m[m["agegrp"] == ag]
    f = smf.glm("hospitalized ~ C(device) + male + C(year) + motor_vehicle + alcohol", data=s,
                family=sm.families.Binomial(), var_weights=s["wn"]).fit(cov_type="cluster", cov_kwds={"groups": s["cl"]})
    for dev in ["E-bike", "E-scooter"]:
        v = f"C(device)[T.{dev}]"
        b_, se = f.params[v], f.bse[v]
        rows.append({"age": lab, "device": dev, "or": np.exp(b_), "lo": np.exp(b_ - 1.96 * se), "hi": np.exp(b_ + 1.96 * se)})
r = pd.DataFrame(rows)
r.to_csv(BASE / "hospitalization_odds_by_age.csv", index=False)
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ages = ["Children (0-17)", "Adults (18-64)", "Older adults (65+)"]
for i, ag in enumerate(ages):
    for j, (dev, col, mk) in enumerate([("E-bike", EBIKE, "o"), ("E-scooter", ESCOOT, "s")]):
        q = r[(r["age"] == ag) & (r["device"] == dev)].iloc[0]
        y = len(ages) - 1 - i + (0.15 if j == 0 else -0.15)
        ax.errorbar(q["or"], y, xerr=[[q["or"] - q["lo"]], [q["hi"] - q["or"]]], fmt=mk, color=col,
                    markersize=4, elinewidth=0.9, capsize=2, label=dev if i == 0 else None)
ax.axvline(1, color=GREY, linewidth=0.8, linestyle="--")
ax.set_yticks(range(len(ages)))
ax.set_yticklabels(ages[::-1])
ax.set_xscale("log")
ax.set_xticks([0.6, 0.8, 1, 1.5, 2])
ax.set_xticklabels(["0.6", "0.8", "1", "1.5", "2"])
ax.set_xlabel("Adjusted odds of hospitalization vs regular bicycle\n(odds ratio, 95% CI, log scale)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
clean(ax)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
fig.tight_layout()
save(fig, "Figure3")
print(r.round(2).to_string(index=False))
