"""DataDelulu engine: differentially private synthetic data + evaluation.

Shared contract used by the Streamlit app:
    preprocess, synthesize, evaluate_utility, membership_attack,
    privacy_frontier, recommend_epsilon, theoretical_attack_bound
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MIN_ROWS = 50
MAX_TARGET_CLASSES = 20
MAX_CATEGORIES = 20


class DataError(ValueError):
    """Raised for user-facing input problems (shown in the app)."""


# --------------------------------------------------------------------------- #
# 1. Validation + preprocessing
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, target: str) -> None:
    if df is None or df.empty:
        raise DataError("The file is empty.")
    if target not in df.columns:
        raise DataError(f"Target column '{target}' not found.")
    if df.shape[1] < 2:
        raise DataError("Need at least one feature column besides the target.")
    if len(df.dropna()) < MIN_ROWS:
        raise DataError(f"Need at least {MIN_ROWS} complete rows (found {len(df.dropna())}).")
    n_classes = df[target].nunique()
    if n_classes < 2:
        raise DataError("Target column has only one value; nothing to predict.")
    if n_classes > MAX_TARGET_CLASSES:
        raise DataError(f"Target has {n_classes} distinct values; choose a categorical target "
                        f"with at most {MAX_TARGET_CLASSES} classes.")


def preprocess(df: pd.DataFrame, target: str, n_bins: int = 8):
    """Clean and discretise the data.

    Returns (encoded_df, meta). encoded_df has every column as small integer codes.
    meta stores how to turn codes back into readable values.

    Assumption (state this if asked): bin edges and category lists are treated as
    public domain knowledge, so they are not charged to the privacy budget.
    """
    validate(df, target)
    df = df.dropna().reset_index(drop=True)
    meta = {"target": target, "columns": {}}
    enc = pd.DataFrame(index=df.index)

    for col in df.columns:
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s) and s.nunique() > 10 and col != target
        if is_numeric:
            lo, hi = float(s.min()), float(s.max())
            if lo == hi:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, n_bins + 1)
            codes = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
            meta["columns"][col] = {"type": "numeric", "edges": edges.tolist(),
                                    "integer": bool(pd.api.types.is_integer_dtype(s))}
            enc[col] = codes
        else:
            s = s.astype(str)
            top = s.value_counts().index[:MAX_CATEGORIES].tolist()
            if len(top) < s.nunique():
                s = s.where(s.isin(top), "Other")
                top = top + ["Other"]
            cats = sorted(set(top))
            mapping = {c: i for i, c in enumerate(cats)}
            meta["columns"][col] = {"type": "categorical", "categories": cats}
            enc[col] = s.map(mapping).astype(int)
    return enc, meta


def domain_size(meta, col):
    c = meta["columns"][col]
    return len(c["edges"]) - 1 if c["type"] == "numeric" else len(c["categories"])


def decode(enc: pd.DataFrame, meta, rng=None) -> pd.DataFrame:
    """Turn integer codes back into readable values (numeric bins -> random value in bin)."""
    rng = rng or np.random.default_rng(0)
    out = pd.DataFrame(index=enc.index)
    for col, info in meta["columns"].items():
        codes = enc[col].to_numpy()
        if info["type"] == "numeric":
            edges = np.array(info["edges"])
            vals = rng.uniform(edges[codes], edges[codes + 1])
            out[col] = np.round(vals).astype(int) if info["integer"] else np.round(vals, 2)
        else:
            cats = np.array(info["categories"])
            out[col] = cats[codes]
    return out


# --------------------------------------------------------------------------- #
# 2. DP synthesizer
# --------------------------------------------------------------------------- #
def _noisy_hist(counts: np.ndarray, scale: float, rng) -> np.ndarray:
    noisy = counts + rng.laplace(0.0, scale, size=counts.shape)
    return np.clip(noisy, 0, None)  # post-processing: free under DP


def synthesize(enc: pd.DataFrame, meta, epsilon: float, n_rows: int | None = None,
               seed: int = 0):
    """Generate DP synthetic data (in encoded form).

    Mechanism: measure the 2-way marginal (feature, target) for every feature with
    the Laplace mechanism, then sample target ~ P(y) and each feature ~ P(x | y).

    Privacy accounting (pure epsilon-DP, add/remove-one-row neighbours):
      * each histogram has L1 sensitivity 1
      * k = number of features measured, each gets epsilon / k
      * sequential composition -> total epsilon
      * everything after measurement is post-processing (no extra cost)
    """
    if not (epsilon > 0 and math.isfinite(epsilon)):
        raise DataError("Epsilon must be a positive number.")
    rng = np.random.default_rng(seed)
    target = meta["target"]
    features = [c for c in enc.columns if c != target]
    k = len(features)
    eps_each = epsilon / k
    scale = 1.0 / eps_each
    ky = domain_size(meta, target)
    y = enc[target].to_numpy()

    cond = {}
    y_counts_est = np.zeros(ky)
    for col in features:
        kx = domain_size(meta, col)
        counts = np.zeros((kx, ky))
        np.add.at(counts, (enc[col].to_numpy(), y), 1)
        noisy = _noisy_hist(counts, scale, rng)
        y_counts_est += noisy.sum(axis=0)
        cond[col] = noisy

    p_y = y_counts_est / k
    p_y = p_y / p_y.sum() if p_y.sum() > 0 else np.full(ky, 1 / ky)
    n = n_rows or len(enc)
    ys = rng.choice(ky, size=n, p=p_y)
    out = {target: ys}
    for col in features:
        table = cond[col] + 1e-9
        table = table / table.sum(axis=0, keepdims=True)
        col_vals = np.empty(n, dtype=int)
        for cls in range(ky):
            idx = np.where(ys == cls)[0]
            if len(idx):
                col_vals[idx] = rng.choice(table.shape[0], size=len(idx), p=table[:, cls])
        out[col] = col_vals

    synth = pd.DataFrame(out)[enc.columns]
    accounting = {
        "mechanism": "Laplace on (feature, target) marginals",
        "epsilon_total": float(epsilon),
        "delta": 0.0,
        "measurements": k,
        "epsilon_per_measurement": eps_each,
        "laplace_scale": scale,
        "composition": "sequential (basic)",
    }
    return synth, accounting


def synthesize_mst(raw_df: pd.DataFrame, epsilon: float, n_rows: int | None = None):
    """Optional: use smartnoise-synth's MST if it is installed.

    Not used by default because smartnoise-synth is hard to install on newer Python.
    """
    from snsynth import Synthesizer  # noqa: import inside so the app works without it
    synth = Synthesizer.create("mst", epsilon=epsilon)
    synth.fit(raw_df, preprocessor_eps=0.0, categorical_columns=list(raw_df.columns))
    return synth.sample(n_rows or len(raw_df))


# --------------------------------------------------------------------------- #
# 3. Evaluation
# --------------------------------------------------------------------------- #
def split(enc: pd.DataFrame, target: str, test_size=0.3, seed=0):
    strat = enc[target] if enc[target].value_counts().min() >= 2 else None
    return train_test_split(enc, test_size=test_size, random_state=seed, stratify=strat)


def _onehot(df, meta, cols):
    parts = []
    for c in cols:
        k = domain_size(meta, c)
        parts.append(np.eye(k)[df[c].to_numpy()])
    return np.hstack(parts)


def evaluate_utility(train: pd.DataFrame, test: pd.DataFrame, synth: pd.DataFrame, meta) -> dict:
    """Train on real vs synthetic, test both on held-out REAL data."""
    target = meta["target"]
    feats = [c for c in train.columns if c != target]
    Xr, yr = _onehot(train, meta, feats), train[target].to_numpy()
    Xs, ys = _onehot(synth, meta, feats), synth[target].to_numpy()
    Xt, yt = _onehot(test, meta, feats), test[target].to_numpy()

    models = {
        "Logistic regression": lambda: LogisticRegression(max_iter=500),
        "Random forest": lambda: RandomForestClassifier(n_estimators=60, max_depth=12,
                                                        random_state=0, n_jobs=-1),
    }
    res = {}
    for name, make in models.items():
        real_acc = make().fit(Xr, yr).score(Xt, yt)
        if len(np.unique(ys)) < 2:
            synth_acc = float(np.mean(yt == ys[0]))  # model can only predict one class
        else:
            synth_acc = make().fit(Xs, ys).score(Xt, yt)
        res[name] = {"real_acc": real_acc, "synth_acc": synth_acc, "gap": real_acc - synth_acc}
    best = max(res, key=lambda m: res[m]["real_acc"])
    baseline = float(np.mean(yt == pd.Series(yr).mode()[0]))  # always guess the majority class
    return {"per_model": res, "baseline_acc": baseline, "real_acc": res[best]["real_acc"],
            "synth_acc": res[best]["synth_acc"], "gap": res[best]["gap"], "model": best}


def membership_attack(members: pd.DataFrame, non_members: pd.DataFrame,
                      synth: pd.DataFrame, n: int = 500, seed: int = 0) -> float:
    """Distance-to-closest-record attack.

    Guess 'member' when a record is unusually close to some synthetic record.
    Returns balanced attack accuracy: 0.5 = pure guessing (ideal privacy).
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(members), len(non_members))
    m = members.sample(n, random_state=seed).to_numpy()
    nm = non_members.sample(n, random_state=seed).to_numpy()
    S = synth.to_numpy()
    if len(S) > 5000:
        S = S[rng.choice(len(S), 5000, replace=False)]

    def dcr(X):
        return np.array([(S != x).sum(axis=1).min() for x in X])  # Hamming distance

    d = np.concatenate([dcr(m), dcr(nm)])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    thr = np.median(d)
    pred = (d < thr).astype(float)
    ties = d == thr
    pred[ties] = rng.random(ties.sum()) < 0.5  # break ties randomly
    return float((pred == labels).mean())


def theoretical_attack_bound(epsilon: float) -> float:
    """Max balanced membership-attack accuracy for any attacker under epsilon-DP."""
    return math.exp(epsilon) / (1 + math.exp(epsilon))


# --------------------------------------------------------------------------- #
# 4. Frontier + budget advisor
# --------------------------------------------------------------------------- #
DEFAULT_EPSILONS = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


def privacy_frontier(enc: pd.DataFrame, meta, epsilons=DEFAULT_EPSILONS, seed=0) -> pd.DataFrame:
    train, test = split(enc, meta["target"], seed=seed)
    rows = []
    for eps in epsilons:
        synth, _ = synthesize(train, meta, eps, seed=seed)
        u = evaluate_utility(train, test, synth, meta)
        rows.append({
            "epsilon": eps,
            "real_acc": u["real_acc"],
            "synth_acc": u["synth_acc"],
            "gap": u["gap"],
            "baseline_acc": u["baseline_acc"],
            "attack_acc": membership_attack(train, test, synth, seed=seed),
            "attack_bound": theoretical_attack_bound(eps),
        })
    return pd.DataFrame(rows)


def recommend_epsilon(frontier: pd.DataFrame, min_acc: float):
    """Smallest epsilon (strongest privacy) whose synthetic accuracy meets the target."""
    ok = frontier[frontier["synth_acc"] >= min_acc]
    if ok.empty:
        return None
    return float(ok.sort_values("epsilon").iloc[0]["epsilon"])
