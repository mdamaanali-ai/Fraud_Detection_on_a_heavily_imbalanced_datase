"""
LEVEL 2 · TASK 3 — Fraud Detection on a heavily imbalanced dataset
Run:  streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Shield", page_icon="🛡️", layout="wide")
T = "plotly_dark"
GOOD, BAD, ACC = "#34d399", "#f87171", "#818cf8"

st.markdown(
    """<style>
.hero{padding:2rem 2.2rem;border-radius:20px;color:#fff;margin-bottom:1.2rem;
background:linear-gradient(135deg,#ef4444 0%,#be185d 50%,#6d28d9 100%);box-shadow:0 10px 30px rgba(190,24,93,.35)}
.hero h1{margin:0;font-size:2.2rem;color:#fff}.hero p{margin:.4rem 0 0;opacity:.92}
div[data-testid="stMetric"]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.22);
padding:14px 16px;border-radius:14px}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>🛡️ Fraud Shield</h1><p>Detecting fraudulent transactions with machine learning — '
    "class imbalance, SMOTE, Precision/Recall/F1/AUC, threshold tuning and scalability.</p></div>",
    unsafe_allow_html=True,
)


# ───────────────────────── data ─────────────────────────
@st.cache_data
def demo_data(n=40000, rate=.012, seed=7):
    X, y = make_classification(n_samples=n, n_features=12, n_informative=8, n_redundant=2,
                               weights=[1 - rate], flip_y=0, class_sep=1.0, random_state=seed)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 13)])
    day = np.array([1, .6, .4, .3, .3, .5, 1, 2, 3, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 4, 4.5, 4.5, 4, 3, 2, 1.5])
    night = np.array([4, 5, 5, 4.5, 4, 3, 2, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1.5, 2, 3, 3.5])
    hour = np.where(y == 1, rng.choice(24, n, p=night / night.sum()), rng.choice(24, n, p=day / day.sum()))
    df["Time"] = rng.integers(0, 2, n) * 86400 + hour * 3600 + rng.integers(0, 3600, n)
    df["Amount"] = np.where(y == 1, rng.lognormal(4.3, 1.5, n), rng.lognormal(3.6, 1.1, n)).round(2)
    df["Class"] = y
    return df


with st.sidebar:
    st.header("⚙️ Settings")
    src = st.radio("Dataset", ["Demo transactions", "Upload creditcard.csv (Kaggle)"])
    if src.startswith("Upload"):
        f = st.file_uploader("Credit Card Fraud Detection CSV", type="csv")
        if f is None:
            st.info("Upload the Kaggle CSV, or switch to the demo data.")
            st.stop()
        df = pd.read_csv(f)
    else:
        df = demo_data()
    cols = df.columns.tolist()
    target = st.selectbox("Fraud label column (1 = fraud)", cols, index=cols.index("Class") if "Class" in cols else len(cols) - 1)
    strategy = st.selectbox("Imbalance technique", ["None (baseline)", "class_weight='balanced'",
                                                    "SMOTE oversampling", "Random undersampling"], index=2)
    max_rows = st.slider("Max rows used for training", 20000, 300000, 80000, 10000,
                         help="All fraud rows are kept; legit rows are down-sampled for speed.")

df = df.copy()
df[target] = df[target].astype(int)
if "Time" in df.columns:
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)

# ───────────────────────── training ─────────────────────────
def subsample(d, tgt, n, seed=42):
    pos, neg = d[d[tgt] == 1], d[d[tgt] == 0]
    if len(neg) > max(n - len(pos), 0):
        neg = neg.sample(max(n - len(pos), 1), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)


@st.cache_data(show_spinner="Training Logistic Regression & Random Forest…")
def train(d, tgt, strategy, seed=42):
    d = d.drop(columns=[c for c in ["Time"] if c in d.columns])
    X, y = d.drop(columns=[tgt]).select_dtypes("number"), d[tgt]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=seed)  # stratified split
    sc = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=X.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=X.columns, index=Xte.index)
    cw = "balanced" if strategy.startswith("class_weight") else None
    Xfit, yfit = Xtr_s, ytr
    if strategy.startswith("SMOTE"):
        from imblearn.over_sampling import SMOTE
        Xfit, yfit = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
    elif strategy.startswith("Random under"):
        from imblearn.under_sampling import RandomUnderSampler
        Xfit, yfit = RandomUnderSampler(random_state=seed).fit_resample(Xtr_s, ytr)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=-1, random_state=seed,
                                                class_weight="balanced_subsample" if cw else None),
    }
    out = {}
    for name, m in models.items():
        m.fit(Xfit, yfit)
        t0 = time.perf_counter()
        proba = m.predict_proba(Xte_s)[:, 1]
        lat = (time.perf_counter() - t0) / len(Xte_s)
        pred = (proba >= .5).astype(int)
        imp = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(m.coef_[0])
        out[name] = dict(proba=proba, pred=pred, latency=lat, imp=imp,
                         precision=precision_score(yte, pred, zero_division=0), recall=recall_score(yte, pred),
                         f1=f1_score(yte, pred), auc=roc_auc_score(yte, proba), pr_auc=average_precision_score(yte, proba),
                         acc=accuracy_score(yte, pred))
    return out, yte, list(X.columns), len(Xfit), int(yfit.sum())


train_df = subsample(df, target, max_rows)
res, yte, feat_names, n_fit, n_fit_pos = train(train_df, target, strategy)
tabs = st.tabs(["⚖️ Imbalance", "🔎 EDA", "🎯 Accuracy Trap", "🤖 Models", "📉 ROC & PR", "🧬 Features", "💰 Threshold", "🚀 Scalability"])

# ───────────────────────── 1 imbalance ─────────────────────────
with tabs[0]:
    n_f = int(df[target].sum())
    c = st.columns(3)
    c[0].metric("Transactions", f"{len(df):,}")
    c[1].metric("Fraudulent", f"{n_f:,}")
    c[2].metric("Fraud percentage", f"{100 * n_f / len(df):.3f}%")
    a, b = st.columns(2)
    cnt = df[target].map({0: "Legit", 1: "Fraud"}).value_counts()
    a.plotly_chart(px.pie(values=cnt.values, names=cnt.index, hole=.55, template=T, color=cnt.index,
                          color_discrete_map={"Legit": GOOD, "Fraud": BAD}), use_container_width=True)
    b.plotly_chart(px.bar(x=cnt.index, y=cnt.values, log_y=True, template=T, color=cnt.index, text=cnt.values,
                          color_discrete_map={"Legit": GOOD, "Fraud": BAD}, labels={"x": "", "y": "count (log scale)"}),
                   use_container_width=True)
    st.warning(f"Only **{100 * n_f / len(df):.3f}%** of transactions are fraud → roughly **1 in {len(df) // max(n_f, 1):,}**. "
               "A model that ignores this will simply learn to say “legit” every time.")

# ───────────────────────── 2 EDA ─────────────────────────
with tabs[1]:
    d = df.assign(Type=df[target].map({0: "Legit", 1: "Fraud"}))
    if "Amount" in d.columns:
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(d[d.Amount < d.Amount.quantile(.99)], x="Amount", color="Type", nbins=60, barmode="overlay",
                                    histnorm="probability density", opacity=.65, template=T, title="Transaction amount (density)",
                                    color_discrete_map={"Legit": GOOD, "Fraud": BAD}), use_container_width=True)
        b.plotly_chart(px.box(d, x="Type", y="Amount", color="Type", log_y=True, template=T, title="Amount by class (log)",
                              color_discrete_map={"Legit": GOOD, "Fraud": BAD}), use_container_width=True)
        s = d.groupby("Type")["Amount"].agg(["mean", "median", "max"]).round(2)
        st.dataframe(s, use_container_width=True)
    if "Hour" in d.columns:
        h = d.groupby(["Hour", "Type"]).size().reset_index(name="n")
        h["share %"] = 100 * h.n / h.groupby("Type").n.transform("sum")
        st.plotly_chart(px.line(h, x="Hour", y="share %", color="Type", markers=True, template=T,
                                title="Time-of-day pattern (share of each class per hour)",
                                color_discrete_map={"Legit": GOOD, "Fraud": BAD}), use_container_width=True)
        peak = h[h.Type == "Fraud"].sort_values("share %").iloc[-1]
        st.info(f"Fraud peaks around **{int(peak.Hour):02d}:00** ({peak['share %']:.1f}% of all fraud).")

# ───────────────────────── 3 accuracy trap ─────────────────────────
with tabs[2]:
    st.subheader("Why accuracy is a misleading metric here")
    base_acc = 1 - yte.mean()
    c = st.columns(3)
    c[0].metric("“Always legit” dummy – accuracy", f"{100 * base_acc:.2f}%")
    c[1].metric("“Always legit” dummy – fraud caught", "0%", delta="recall = 0", delta_color="inverse")
    c[2].metric("Random Forest – recall", f"{100 * res['Random Forest']['recall']:.1f}%")
    st.markdown(f"""
A classifier that **never** flags fraud scores **{100 * base_acc:.2f}% accuracy** on the test set — yet it catches **zero** criminals.
Accuracy rewards getting the majority class right, so it hides failure on the minority class we actually care about.

**Use instead:** *Precision* (how many alerts are real), *Recall* (how much fraud we catch), *F1* (balance of both),
and *ROC-AUC / PR-AUC* (ranking quality independent of threshold).
""")

# ───────────────────────── 4 models ─────────────────────────
with tabs[3]:
    st.caption(f"Technique: **{strategy}** · training rows after balancing: {n_fit:,} ({100 * n_fit_pos / n_fit:.1f}% fraud) · "
               "test set is untouched & stratified (fraud appears in both splits).")
    tbl = pd.DataFrame({m: {"Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["auc"],
                            "PR-AUC": r["pr_auc"], "Accuracy": r["acc"]} for m, r in res.items()}).T
    st.dataframe(tbl.style.format("{:.4f}").highlight_max(axis=0, color="#065f46"), use_container_width=True)
    long = tbl.reset_index().melt("index", var_name="metric", value_name="score")
    st.plotly_chart(px.bar(long, x="metric", y="score", color="index", barmode="group", template=T, text_auto=".2f",
                           labels={"index": "model"}), use_container_width=True)
    cm_cols = st.columns(2)
    for col, (m, r) in zip(cm_cols, res.items()):
        cm = confusion_matrix(yte, r["pred"])
        col.plotly_chart(px.imshow(cm, text_auto=True, template=T, x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"],
                                   color_continuous_scale="Purples", title=m), use_container_width=True)
    st.markdown("""
**Which metric matters most?** For fraud, **Recall** is usually the priority — a missed fraud (false negative) costs the full
transaction amount plus chargebacks. But recall alone is gameable (flag everything), so we watch **Precision** too: every false alarm
annoys a genuine customer and costs analyst time. The **F1 / PR-AUC** summarise this trade-off; the *Threshold* tab lets you pick the
operating point using actual costs.
""")

# ───────────────────────── 5 curves ─────────────────────────
with tabs[4]:
    a, b = st.columns(2)
    fr, fp = go.Figure(), go.Figure()
    for (m, r), colr in zip(res.items(), [ACC, "#f472b6"]):
        f_, t_, _ = roc_curve(yte, r["proba"])
        fr.add_trace(go.Scatter(x=f_, y=t_, name=f"{m} (AUC {r['auc']:.3f})", line=dict(color=colr, width=3)))
        p_, rc_, _ = precision_recall_curve(yte, r["proba"])
        fp.add_trace(go.Scatter(x=rc_, y=p_, name=f"{m} (AP {r['pr_auc']:.3f})", line=dict(color=colr, width=3)))
    fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="chance"))
    fr.update_layout(template=T, title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    fp.update_layout(template=T, title="Precision–Recall curve", xaxis_title="Recall", yaxis_title="Precision")
    a.plotly_chart(fr, use_container_width=True)
    b.plotly_chart(fp, use_container_width=True)
    st.caption("With heavy imbalance the PR curve is more informative than ROC: ROC can look excellent even when precision is poor.")

# ───────────────────────── 6 features ─────────────────────────
with tabs[5]:
    m = st.radio("Model", list(res), horizontal=True)
    imp = pd.Series(res[m]["imp"], index=feat_names).sort_values().tail(15)
    st.plotly_chart(px.bar(imp, orientation="h", template=T, color=imp, color_continuous_scale="Sunset",
                           labels={"value": "importance" if m == "Random Forest" else "|coefficient| (scaled)", "index": ""})
                    .update_layout(coloraxis_showscale=False), use_container_width=True)
    st.caption(f"Top driver: **{imp.index[-1]}**. Always sanity-check importances with domain experts — "
               "importance ≠ causation.")

# ───────────────────────── 7 threshold ─────────────────────────
with tabs[6]:
    m = st.selectbox("Model", list(res), index=1, key="thr_model")
    p = res[m]["proba"]
    c1, c2, c3 = st.columns(3)
    thr = c1.slider("Decision threshold", .01, .99, .5, .01)
    cost_fn = c2.number_input("Cost of a missed fraud (FN)", 1.0, 100000.0, 200.0)
    cost_fp = c3.number_input("Cost of a false alarm (FP)", .1, 10000.0, 5.0)
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(yte, pred)
    tn, fp_, fn, tp = cm.ravel()
    k = st.columns(4)
    k[0].metric("Precision", f"{precision_score(yte, pred, zero_division=0):.3f}")
    k[1].metric("Recall", f"{recall_score(yte, pred):.3f}")
    k[2].metric("F1", f"{f1_score(yte, pred):.3f}")
    k[3].metric("Total cost", f"{fn * cost_fn + fp_ * cost_fp:,.0f}")
    grid = np.linspace(.01, .99, 99)
    rows = []
    for g in grid:
        pr = (p >= g).astype(int)
        t_n, f_p, f_n, t_p = confusion_matrix(yte, pr, labels=[0, 1]).ravel()
        rows.append((g, precision_score(yte, pr, zero_division=0), recall_score(yte, pr), f_n * cost_fn + f_p * cost_fp))
    g = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "cost"])
    a, b = st.columns(2)
    a.plotly_chart(px.line(g, x="threshold", y=["precision", "recall"], template=T, title="Precision–Recall trade-off"), use_container_width=True)
    best = g.loc[g.cost.idxmin()]
    fig = px.line(g, x="threshold", y="cost", template=T, title="Business cost vs threshold")
    fig.add_vline(x=best.threshold, line_dash="dash", line_color=GOOD)
    b.plotly_chart(fig, use_container_width=True)
    st.success(f"Cheapest threshold with these costs: **{best.threshold:.2f}** (total cost {best.cost:,.0f}). "
               "Raising the threshold → fewer false alarms but more missed fraud; lowering it does the opposite.")

# ───────────────────────── 8 scalability ─────────────────────────
with tabs[7]:
    st.subheader("Could this handle 1 million transactions per hour?")
    need = 1_000_000 / 3600
    c = st.columns(3)
    c[0].metric("Required throughput", f"{need:,.0f} tx/sec")
    for col, (m, r) in zip(c[1:], res.items()):
        col.metric(f"{m} (measured, batch)", f"{1 / r['latency']:,.0f} tx/sec")
    st.markdown("""
**Short answer: yes for scoring, with the right architecture.** 1 M/hour ≈ 278 transactions/second — modest for a tree/linear model.

- **Inference is cheap, features are the bottleneck.** Real-time features (e.g. “transactions by this card in last 10 min”) need a low-latency
  feature store such as Redis. Pre-compute rolling aggregates instead of querying raw history.
- **Scale horizontally.** Serve the model behind a stateless API (FastAPI / TorchServe / SageMaker) and autoscale replicas;
  use Kafka or Kinesis to stream events, and micro-batch where latency allows.
- **Keep latency budget < 100 ms.** Prefer Logistic Regression, gradient-boosted trees or a distilled model; avoid heavy ensembles online.
- **Model drift.** Fraudsters adapt — monitor precision/recall and data drift, retrain regularly (weekly/daily) and use champion–challenger testing.
- **Delayed labels.** Confirmed fraud arrives days later (chargebacks) — plan for label delay and human-in-the-loop review of borderline scores.
- **Never leak:** SMOTE / scaling must be fitted on the training split only (done here).
""")
