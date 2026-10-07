# DataDelulu — Looks real, reveals nothing
Team: The Missing Bit · Hacksphere 2026 · AI/ML Problem Statement #5

Generates differentially private (ε-DP) synthetic tabular data, measures how useful
it is (accuracy) and how safe it is (membership-inference attack), maps the full
privacy–utility frontier, and recommends the strongest privacy for a target accuracy.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
pytest -q            # 10 tests
```
Optional real dataset: `python data/download_adult.py` → upload `data/adult.csv`, target `income`.
`data/loan_demo.csv` is a generated sample dataset (not real people), made by `data/make_demo_data.py`.

## Deploy (Streamlit Community Cloud)
Push this folder to a public GitHub repo → share.streamlit.io → New app →
main file `app/streamlit_app.py` → Deploy.

## Project layout
| Path | What |
|---|---|
| `src/engine.py` | Preprocess, DP synthesizer, privacy accounting, utility test, attack, frontier, advisor |
| `app/streamlit_app.py` | Web app (upload → ε/advisor → synthetic data, charts, report) |
| `tests/test_engine.py` | Schema, budget accounting, bad input, ranges, advisor |

## How the privacy works
1. Numeric columns are binned; categories capped at 20 (bins/categories treated as public).
2. For each feature, the (feature, target) count table gets Laplace noise.
   k features → each gets ε/k; sequential composition gives total ε (pure DP, δ = 0).
3. Synthetic rows are sampled from the noisy tables (post-processing, no extra privacy cost).

## Why zero utility loss is impossible
If synthetic data reproduced every count exactly, adding or removing one person changes some
count with certainty, so the output reveals that person — violating ε-DP for every finite ε
(and every δ < 1). The minimum error on a counting query is Θ(1/ε), matched by Laplace.
For any attacker, membership-attack accuracy ≤ e^ε / (1 + e^ε); it equals 50% only as ε → 0.

## Known limits (be upfront with judges)
- The synthesizer models each feature conditioned on the target (naive-Bayes structure),
  so feature-to-feature correlations are not preserved. Upgrade path: MST
  (`synthesize_mst` in engine.py, needs `pip install smartnoise-synth`, untested on Python 3.13).
- Tabular data only; text and time-series are future work.
