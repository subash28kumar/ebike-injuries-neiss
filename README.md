# E-bike injuries among U.S. children, 2019–2025

Code for the paper *"E-bike injuries among U.S. children surged in 2025: national estimates from emergency room reports, 2019–2025, using AI to read injury stories"* by Subash Kumar (ORCID [0009-0003-6661-3424](https://orcid.org/0009-0003-6661-3424)).

The study uses the U.S. Consumer Product Safety Commission's National Electronic Injury Surveillance System (NEISS), 2019–2025. CPSC changed its product codes in 2024: before then, e-bikes shared one code with mopeds. A text classifier trained on the 2024–2025 injury stories, where e-bikes and mopeds have separate codes, sorts the older shared-code records. This gives one consistent e-bike series across the code change.

## Data

Download the annual NEISS files for 2019 to 2025 (tab-separated) from https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data and save them in a folder called `data/` as `neiss2019.tsv` … `neiss2025.tsv`.

## How to run

```
pip install -r requirements.txt
python neiss_extract.py    # keep bike, scooter and similar injuries from all 7 years
python neiss_build.py      # label each injury by device (codes + injury story + text classifier)
python neiss_analysis.py   # national estimates with survey 95% CIs; hospitalization odds
python neiss_figures.py    # Figures 1-3 and A1
```

## Main results

- Estimated U.S. emergency room visits for e-bike injuries rose from about 5,100 in 2019 to about 101,300 in 2025.
- Children's e-bike visits rose from about 600 in 2019 to about 25,600 in 2025, a quarter of all e-bike injuries.
- Injured children on e-bikes had 1.68 times the odds of hospital admission or transfer compared with children injured on regular bicycles (95% CI 1.25–2.26).
- The classifier separated e-bikes from mopeds with a cross-validated AUC of 0.995, and the e-bike estimates match CPSC's published figures for 2022–2024.

## How to cite

See `CITATION.cff`, or use the "Cite this repository" button on GitHub.

## License

MIT
