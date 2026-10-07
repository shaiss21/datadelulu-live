import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine import (DataError, membership_attack, preprocess, recommend_epsilon,  # noqa: E402
                        split, synthesize, theoretical_attack_bound, evaluate_utility)

DATA = Path(__file__).resolve().parents[1] / "data" / "loan_demo.csv"


@pytest.fixture(scope="module")
def prepared():
    df = pd.read_csv(DATA)
    return preprocess(df, "defaulted")


def test_schema_matches(prepared):
    enc, meta = prepared
    synth, _ = synthesize(enc, meta, 1.0)
    assert list(synth.columns) == list(enc.columns)
    assert len(synth) == len(enc)


def test_codes_stay_in_domain(prepared):
    enc, meta = prepared
    synth, _ = synthesize(enc, meta, 0.5)
    from src.engine import domain_size
    for c in synth.columns:
        assert synth[c].between(0, domain_size(meta, c) - 1).all()


def test_budget_accounting_adds_up(prepared):
    enc, meta = prepared
    _, acc = synthesize(enc, meta, 2.0)
    assert acc["epsilon_per_measurement"] * acc["measurements"] == pytest.approx(2.0)


@pytest.mark.parametrize("eps", [0, -1, float("inf")])
def test_invalid_epsilon_rejected(prepared, eps):
    enc, meta = prepared
    with pytest.raises(DataError):
        synthesize(enc, meta, eps)


def test_bad_inputs_rejected():
    with pytest.raises(DataError):
        preprocess(pd.DataFrame(), "x")
    with pytest.raises(DataError):
        preprocess(pd.DataFrame({"a": range(10), "y": [0, 1] * 5}), "y")  # too few rows
    with pytest.raises(DataError):
        preprocess(pd.DataFrame({"a": range(100), "y": [1] * 100}), "y")  # single class


def test_attack_and_utility_ranges(prepared):
    enc, meta = prepared
    train, test = split(enc, "defaulted")
    synth, _ = synthesize(train, meta, 1.0)
    assert 0.0 <= membership_attack(train, test, synth, n=200) <= 1.0
    u = evaluate_utility(train, test, synth, meta)
    assert 0.0 <= u["synth_acc"] <= 1.0


def test_attack_bound():
    assert theoretical_attack_bound(1e-9) == pytest.approx(0.5)
    assert theoretical_attack_bound(1.0) == pytest.approx(np.e / (1 + np.e))


def test_recommend_epsilon():
    f = pd.DataFrame({"epsilon": [0.1, 1, 10], "synth_acc": [0.6, 0.75, 0.8]})
    assert recommend_epsilon(f, 0.7) == 1
    assert recommend_epsilon(f, 0.9) is None
