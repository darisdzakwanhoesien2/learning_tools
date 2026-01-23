import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Nearest Mean & KNN Classifier", layout="wide")
st.title("📊 Nearest Mean and K-Nearest Neighbours Classification")

# =====================================================
# DATA
# =====================================================
C1 = np.array([
    [1, 2, 2, 2, 3, 10, 15, 16, 16, 17],
    [2, 1, 2, 3, 2,  2,  1,  2,  3,  2]
], dtype=float).T

C2 = np.array([
    [6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10],
    [1, 3, 2, 1, 2, 3, 1, 3, 1, 3,  3]
], dtype=float).T

x = np.array([9.6, 2.0])

# =====================================================
# HELPERS
# =====================================================
def euclidean(a, b):
    return np.linalg.norm(a - b)

# =====================================================
# DISPLAY DATA
# =====================================================
st.header("📥 Given Data")

c1_col, c2_col = st.columns(2)

with c1_col:
    st.subheader("Class C₁ Points")
    st.write(C1)

with c2_col:
    st.subheader("Class C₂ Points")
    st.write(C2)

st.subheader("Test Point")
st.write(x)

# =====================================================
# PART (a) — NEAREST MEAN
# =====================================================
st.divider()
st.header("🧮 (a) Nearest Mean Classifier")

mu1 = C1.mean(axis=0)
mu2 = C2.mean(axis=0)

d_mu1 = euclidean(x, mu1)
d_mu2 = euclidean(x, mu2)

st.subheader("Class Centroids")
st.write("μ₁ =", mu1)
st.write("μ₂ =", mu2)

st.subheader("Distances to Centroids")
st.write("d(x, μ₁) =", round(d_mu1, 4))
st.write("d(x, μ₂) =", round(d_mu2, 4))

if d_mu1 < d_mu2:
    st.success("✅ Nearest Mean Classification → Class C₁")
else:
    st.success("✅ Nearest Mean Classification → Class C₂")

# =====================================================
# PART (b) — NEAREST NEIGHBOUR (1-NN)
# =====================================================
st.divider()
st.header("📍 (b) Nearest Neighbour Classifier (1-NN)")

distances = []

for i, p in enumerate(C1):
    distances.append(("C1", i, euclidean(x, p)))

for i, p in enumerate(C2):
    distances.append(("C2", i, euclidean(x, p)))

distances_sorted = sorted(distances, key=lambda t: t[2])

st.subheader("All Distances (sorted)")
st.write(distances_sorted)

nearest = distances_sorted[0]
st.success(f"✅ Nearest neighbour belongs to {nearest[0]}")

# =====================================================
# PART (c) — KNN (K=3)
# =====================================================
st.divider()
st.header("📌 (c) K-Nearest Neighbours (K = 3)")

K = 3
knn = distances_sorted[:K]

st.subheader("3 Nearest Neighbours")
st.write(knn)

votes = {"C1": 0, "C2": 0}
for item in knn:
    votes[item[0]] += 1

st.subheader("Voting Result")
st.write(votes)

if votes["C1"] > votes["C2"]:
    st.success("✅ KNN Classification → Class C₁")
else:
    st.success("✅ KNN Classification → Class C₂")

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Visualization")

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(C1[:,0], C1[:,1], label="Class C₁", s=80)
ax.scatter(C2[:,0], C2[:,1], label="Class C₂", s=80)
ax.scatter(x[0], x[1], color="black", marker="x", s=120, label="Test point x")

ax.scatter(mu1[0], mu1[1], marker="*", s=200, label="Centroid μ₁")
ax.scatter(mu2[0], mu2[1], marker="*", s=200, label="Centroid μ₂")

# Draw lines to 3 nearest neighbours
for item in knn:
    label, idx, _ = item
    if label == "C1":
        p = C1[idx]
    else:
        p = C2[idx]
    ax.plot([x[0], p[0]], [x[1], p[1]], "--", alpha=0.5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Nearest Mean and KNN Classification")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.caption("🚀 Nearest Mean, 1-NN and KNN classification demo.")
