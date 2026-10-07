"""DataDelulu - Looks real, reveals nothing.  Run:  streamlit run app/streamlit_app.py"""
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine import (DataError, DEFAULT_EPSILONS, decode, evaluate_utility,  # noqa: E402
                        membership_attack, preprocess, privacy_frontier,
                        recommend_epsilon, split, synthesize, theoretical_attack_bound)

st.set_page_config(page_title="DataDelulu", page_icon="🕵️", layout="wide")
st.title("DataDelulu")
st.caption("Looks real, reveals nothing — differentially private synthetic data with measured trade-offs")


# ----------------------------- cached steps -------------------------------- #
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def run_preprocess(df, target):
    return preprocess(df, target)


@st.cache_data(show_spinner=False)
def run_frontier(enc, meta):
    return privacy_frontier(enc, meta)


# ----------------------------- sidebar: input ------------------------------ #
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Demo loan dataset", "Upload CSV"])
    df = None
    try:
        if source == "Upload CSV":
            up = st.file_uploader("CSV file", type="csv")
            if up is not None:
                df = load_csv(up.getvalue())
        else:
            df = pd.read_csv(ROOT / "data" / "loan_demo.csv")
    except Exception as e:  # unreadable / malformed file
        st.error(f"Could not read the file: {e}")

if df is None:
    st.info("Upload a CSV (or pick the demo dataset) to begin.")
    st.stop()

with st.sidebar:
    cols = list(df.columns)
    default_target = cols.index("defaulted") if "defaulted" in cols else len(cols) - 1
    target = st.selectbox("Column to predict (target)", cols, index=default_target)

    st.header("2. Privacy")
    mode = st.radio("Choose by", ["Privacy level (ε)", "Target accuracy (advisor)"])

try:
    enc, meta = run_preprocess(df, target)
except DataError as e:
    st.error(str(e))
    st.stop()

with st.spinner("Mapping the privacy–utility frontier…"):
    frontier = run_frontier(enc, meta)

with st.sidebar:
    if mode == "Privacy level (ε)":
        epsilon = st.select_slider("ε (smaller = more private)", options=list(DEFAULT_EPSILONS), value=1.0)
    else:
        lo, hi = float(frontier.synth_acc.min()), float(frontier.synth_acc.max())
        min_acc = st.slider("Minimum accuracy needed", 0.0, 1.0, round(lo, 2), 0.01, format="%.2f")
        rec = recommend_epsilon(frontier, min_acc)
        if rec is None:
            st.warning(f"No tested ε reaches {min_acc:.0%}. Best achievable is {hi:.1%}. Using ε = 10.")
            epsilon = 10.0
        else:
            st.success(f"Recommended ε = {rec} (strongest privacy meeting {min_acc:.0%})")
            epsilon = rec

# ----------------------------- generate ------------------------------------ #
train, test = split(enc, target)
synth_enc, accounting = synthesize(train, meta, epsilon)
util = evaluate_utility(train, test, synth_enc, meta)
attack = membership_attack(train, test, synth_enc)
bound = theoretical_attack_bound(epsilon)
synth_df = decode(synth_enc, meta)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Privacy budget ε", f"{epsilon}")
c2.metric("Accuracy: real → synthetic", f"{util['synth_acc']:.1%}",
          f"{-util['gap']:+.1%} vs real ({util['real_acc']:.1%})")
c3.metric("Membership attack success", f"{attack:.1%}", "50% = pure guessing", delta_color="off")
c4.metric("Theoretical attack limit", f"{bound:.1%}", help="Max for ANY attacker under ε-DP: e^ε/(1+e^ε)")

tab_data, tab_frontier, tab_compare, tab_report = st.tabs(
    ["Synthetic data", "Privacy–utility frontier", "Real vs synthetic", "Privacy report"])

with tab_data:
    st.subheader("Synthetic rows (no real person in here)")
    st.dataframe(synth_df.head(50), width="stretch")
    st.download_button("Download synthetic CSV", synth_df.to_csv(index=False).encode(),
                       f"datadelulu_synthetic_eps{epsilon}.csv", "text/csv")

with tab_frontier:
    st.subheader("What accuracy does each privacy level cost?")
    long = frontier.melt("epsilon", ["real_acc", "synth_acc", "baseline_acc"], "series", "accuracy")
    long["series"] = long["series"].map({"real_acc": "Trained on real data",
                                         "synth_acc": "Trained on synthetic data",
                                         "baseline_acc": "Majority-class baseline"})
    acc_chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("epsilon:Q", scale=alt.Scale(type="log"), title="ε (log scale)"),
        y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False), title="Test accuracy on real data"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(
            domain=["Trained on real data", "Trained on synthetic data", "Majority-class baseline"],
            range=["#2B3A67", "#0F766E", "#B0B4C3"])))
    rule = alt.Chart(pd.DataFrame({"epsilon": [epsilon]})).mark_rule(strokeDash=[4, 4]).encode(x="epsilon:Q")
    st.altair_chart(acc_chart + rule, width="stretch")

    st.subheader("How much does an attacker learn?")
    att = frontier.melt("epsilon", ["attack_acc", "attack_bound"], "series", "success")
    att["series"] = att["series"].map({"attack_acc": "Our measured attack",
                                       "attack_bound": "Theoretical max (any attacker)"})
    att_chart = alt.Chart(att).mark_line(point=True).encode(
        x=alt.X("epsilon:Q", scale=alt.Scale(type="log"), title="ε (log scale)"),
        y=alt.Y("success:Q", scale=alt.Scale(domain=[0.4, 1.0]), title="Attack success"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(
            domain=["Our measured attack", "Theoretical max (any attacker)"],
            range=["#0F766E", "#B4533D"])))
    st.altair_chart(att_chart + rule, width="stretch")
    st.dataframe(frontier.round(3), width="stretch", hide_index=True)

with tab_compare:
    st.subheader("Do the distributions match?")
    real_df = decode(train, meta)
    col = st.selectbox("Column", list(meta["columns"]))
    info = meta["columns"][col]
    if info["type"] == "numeric":
        r, s = pd.cut(real_df[col], info["edges"], include_lowest=True), pd.cut(synth_df[col], info["edges"], include_lowest=True)
        r, s = r.astype(str), s.astype(str)
    else:
        r, s = real_df[col], synth_df[col]
    comp = pd.concat([r.value_counts(normalize=True).rename("Real"),
                      s.value_counts(normalize=True).rename("Synthetic")], axis=1).fillna(0)
    comp = comp.reset_index(names="value").melt("value", var_name="data", value_name="share")
    st.altair_chart(alt.Chart(comp).mark_bar().encode(
        x=alt.X("value:N", title=col), xOffset="data:N", y=alt.Y("share:Q", axis=alt.Axis(format="%")),
        color="data:N"), width="stretch")

with tab_report:
    per_model = "\n".join(f"| {m} | {v['real_acc']:.1%} | {v['synth_acc']:.1%} | {v['gap']:+.1%} |"
                          for m, v in util["per_model"].items())
    report = f"""# DataDelulu privacy report
Generated: {datetime.now():%Y-%m-%d %H:%M}

## Privacy guarantee
- Mechanism: {accounting['mechanism']}
- Total privacy budget: ε = {accounting['epsilon_total']}, δ = {accounting['delta']}
- Measurements: {accounting['measurements']} (ε = {accounting['epsilon_per_measurement']:.4f} each, Laplace scale {accounting['laplace_scale']:.2f})
- Composition: {accounting['composition']}; sampling is post-processing (no extra cost)

## Utility (models trained on synthetic data, tested on real held-out data)
| Model | Real-trained | Synthetic-trained | Gap |
|---|---|---|---|
{per_model}

Majority-class baseline: {util['baseline_acc']:.1%}

## Privacy test
- Membership-inference attack (distance to closest record): {attack:.1%} success (50% = guessing)
- Theoretical maximum for any attacker at this ε: {bound:.1%}

## Why zero loss is impossible
If synthetic data reproduced every count exactly, adding or removing one person would change
some count deterministically, revealing that person and violating ε-DP for every finite ε.
Minimum error on a counting query is of order 1/ε; the Laplace mechanism achieves it.

## Assumptions
Column bins and category lists are treated as public knowledge (not charged to ε).
Rows: {len(train)} real (training) → {len(synth_df)} synthetic.
Supports compliance discussions under data-protection laws such as India's DPDP Act; not legal advice.
"""
    st.markdown(report)
    st.download_button("Download report (.md)", report.encode(), f"datadelulu_report_eps{epsilon}.md")
