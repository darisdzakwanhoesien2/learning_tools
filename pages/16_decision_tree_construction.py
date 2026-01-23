import streamlit as st
import pandas as pd
import numpy as np
from math import log2

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Automatic Decision Tree Construction",
    layout="wide"
)
st.title("🌳 Automatic Construction of a Decision Tree")

# =====================================================
# DATASET
# =====================================================
data = pd.DataFrame({
    "a": [1, 1, 0, 1],
    "b": [0, 1, 1, 1],
    "c": [1, 0, 1, 1],
    "d": [1, 0, 0, 1],
    "Class": ["I", "II", "II", "I"]
})

st.header("📥 Training Data")
st.dataframe(data)

# =====================================================
# ENTROPY + INFORMATION GAIN
# =====================================================
def entropy(labels):
    counts = labels.value_counts()
    total = len(labels)
    H = 0.0
    for c in counts:
        p = c / total
        H -= p * log2(p)
    return H

def information_gain(df, feature, target="Class"):
    H_total = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0.0

    for v in values:
        subset = df[df[feature] == v]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset[target])

    return H_total - weighted_entropy

# =====================================================
# PART (a) — TREE STARTING FROM FEATURE a
# =====================================================
st.divider()
st.header("🅰️ (a) Tree Starting from Feature a")

def build_tree_fixed_root(df, root_feature):
    tree = {}
    for value in sorted(df[root_feature].unique()):
        subset = df[df[root_feature] == value]
        majority = subset["Class"].mode()[0]
        tree[value] = majority
    return tree

tree_a = build_tree_fixed_root(data, "a")

st.subheader("Decision Rule (Root = a)")
st.write(tree_a)

st.markdown("""
Interpretation:

- If **a = 0** → predict class **II**
- If **a = 1** → predict class **I**
""")

# =====================================================
# PART (b) — INFORMATION GAIN TREE
# =====================================================
st.divider()
st.header("📊 (b) Tree Using Information Gain")

features = ["a", "b", "c", "d"]
IG = {}

for f in features:
    IG[f] = information_gain(data, f)

ig_table = pd.DataFrame({
    "Feature": IG.keys(),
    "Information Gain": IG.values()
}).sort_values("Information Gain", ascending=False)

st.subheader("Information Gain Table")
st.dataframe(ig_table)

best_feature = ig_table.iloc[0]["Feature"]

st.success(f"Best root feature according to Information Gain → **{best_feature}**")

tree_ig = build_tree_fixed_root(data, best_feature)

st.subheader("Decision Rule (Root = Information Gain)")
st.write(tree_ig)

# =====================================================
# PART (c) — COMPARISON
# =====================================================
st.divider()
st.header("🧠 (c) Difference Between the Trees")

st.markdown(f"""
### 🌲 Tree 1 (Fixed Root = a)
- Root feature is manually chosen.
- Does not consider class purity or entropy reduction.
- May lead to suboptimal splits.

### 🌲 Tree 2 (Information Gain Root = {best_feature})
- Root is selected automatically based on entropy reduction.
- Produces a more informative split.
- Usually leads to smaller trees and better classification accuracy.

### ✅ Why they differ
Because **information gain measures how well a feature separates the classes**.
A manually chosen feature may not maximize class separation.

Automatic feature selection leads to more optimal decision boundaries.
""")

# =====================================================
# OPTIONAL: INTERACTIVE CLASSIFICATION
# =====================================================
st.divider()
st.header("🎯 Test a Sample")

a = st.selectbox("a", [0, 1])
b = st.selectbox("b", [0, 1])
c = st.selectbox("c", [0, 1])
d = st.selectbox("d", [0, 1])

sample = {"a": a, "b": b, "c": c, "d": d}

# Prediction using fixed tree
pred_a = tree_a[a]

# Prediction using IG tree
root = best_feature
pred_ig = tree_ig[sample[root]]

c1, c2 = st.columns(2)

with c1:
    st.subheader("Prediction (Root = a)")
    st.success(pred_a)

with c2:
    st.subheader("Prediction (Information Gain Root)")
    st.success(pred_ig)

st.caption("🚀 Decision Tree construction demo using entropy and information gain.")
