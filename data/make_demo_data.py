"""Creates data/loan_demo.csv: a GENERATED sample dataset for demos/testing.

It is not real customer data. For a real dataset run data/download_adult.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 6000
age = rng.integers(21, 66, n)
education = rng.choice(["High school", "Bachelors", "Masters", "PhD"], n, p=[.4, .35, .2, .05])
edu_boost = pd.Series(education).map({"High school": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}).to_numpy()
employment = rng.choice(["Salaried", "Self-employed", "Unemployed"], n, p=[.7, .22, .08])
income = np.round(rng.lognormal(10.3 + 0.15 * edu_boost + 0.005 * (age - 21), 0.4)).astype(int)
income[employment == "Unemployed"] //= 4
credit_score = np.clip(rng.normal(620 + 0.0009 * income + 2 * (age - 21), 60), 300, 900).astype(int)
loan_amount = np.round(rng.uniform(0.2, 1.5, n) * income).astype(int)
city = rng.choice(["Hyderabad", "Bengaluru", "Chennai", "Vijayawada", "Guntur"], n)
logit = (-0.012 * (credit_score - 650) + 0.9 * (loan_amount / income) - 0.4 * edu_boost
         + 1.4 * (employment == "Unemployed") - 1.2)
default = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
df = pd.DataFrame({"age": age, "education": education, "employment": employment,
                   "annual_income": income, "credit_score": credit_score,
                   "loan_amount": loan_amount, "city": city, "defaulted": default})
out = Path(__file__).parent / "loan_demo.csv"
df.to_csv(out, index=False)
print(f"wrote {out} {df.shape}, default rate {np.mean(default == 'Yes'):.1%}")
