"""
Paper 4, step 1: pull bicycle, e-bike, scooter, e-scooter and similar injuries out of the NEISS files.

Save the CPSC annual files neiss2019.tsv ... neiss2025.tsv in a folder called data/ and run:
    python neiss_extract.py

It keeps every injury whose description mentions a bike, scooter, hoverboard, skateboard,
moped, unicycle or similar (powered or not), from all 7 years, and saves:
    data/neiss_micromobility_2019_2025.csv
"""
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / "data"
PATTERN = re.compile(
    r"BIKE|BICYCLE|BICYCLIST|CYCLIST|SCOOTER|HOVERBOARD|HOVER BOARD|SKATEBOARD|SKATE BOARD|"
    r"MOPED|UNICYCLE|ONEWHEEL|ONE WHEEL|SEGWAY|PEDAL ASSIST|E-?BIKE|EBIKE|E-?SCOOTER|ESCOOTER",
    re.IGNORECASE,
)

parts = []
for year in range(2019, 2026):
    f = BASE / f"neiss{year}.tsv"
    if not f.exists():
        print(f"Missing {f.name}, skipping")
        continue
    df = pd.read_csv(f, sep="\t", dtype=str, encoding="latin-1", low_memory=False)
    df.columns = [c.replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
    narr_col = next(c for c in df.columns if c.lower() == "narrative" or c.lower().startswith("narr"))
    keep = df[narr_col].fillna("").str.contains(PATTERN)
    sub = df[keep].copy()
    sub["file_year"] = year
    parts.append(sub)
    print(f"{year}: {len(df):,} injuries in file, {len(sub):,} kept")

out = pd.concat(parts, ignore_index=True)
csv_path = BASE / "neiss_micromobility_2019_2025.csv"
out.to_csv(csv_path, index=False)
print(f"Saved {csv_path.name} ({len(out):,} rows)")
