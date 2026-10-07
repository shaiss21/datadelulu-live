"""Downloads the UCI Adult Income dataset to data/adult.csv (target column: income)."""
import pandas as pd
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"]
df = pd.read_csv(URL, names=COLS, skipinitialspace=True, na_values="?")
df = df.drop(columns=["fnlwgt", "education_num"])
out = Path(__file__).parent / "adult.csv"
df.to_csv(out, index=False)
print(f"wrote {out} {df.shape}")
