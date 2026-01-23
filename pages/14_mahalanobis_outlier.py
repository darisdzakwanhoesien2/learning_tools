import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Mahalanobis Distance & Outlier Detection", layout="wide")
st.title("📏 Mahalanobis Distance and Outlier Detection")

# =====================================================
# DATA
# =====================================================
# Each column is a sample [weight, height]^T
C = np.array([
    [3,  8, 12, 17, 15, 11,  6,  9, 13, 10,  6, 16],
    [20,40, 60, 85, 70, 50, 25, 55, 70, 40, 35, 80]
], dtype=float).T

# New sample
y = np.array([7.0, 65.0])

# =====================================================
# HELPERS
# =====================================================
def mahalanobis_distance(x, mu, Sigma_inv):
    d = x - mu
    return float(np.sqrt(d.T @ Sigma_inv @ d))

# =====================================================
# DISPLAY DATA
# =====================================================
st.header("📥 Given Dataset")

st.markdown("Each row is a sample: **[weight, height]**")
st.write(C)

st.subheader("New Sample")
st.write(y)

# =====================================================
# PART (a) — WHY MAHALANOBIS
# =====================================================
st.divider()
st.header("🧠 (a) Why Mahalanobis Distance?")

st.markdown("""
**Euclidean distance**
- Treats all dimensions equally.
- Ignores correlation between features.
- Sensitive to scale differences (e.g., weight vs height).

**Mahalanobis distance**
- Accounts for **variance and correlation** of the data.
- Automatically scales dimensions.
- Measures distance in units of standard deviation.
- Much better for **outlier detection** and statistical modeling.

👉 Therefore, Mahalanobis distance is preferred when features have different scales or correlations.
""")

# =====================================================
# PART (b) — MEAN AND COVARIANCE
# =====================================================
st.divider()
st.header("🧮 (b) Compute Mean and Covariance")

mu = C.mean(axis=0)
Sigma = np.cov(C, rowvar=False, bias=False)
Sigma_inv = np.linalg.inv(Sigma)

st.subheader("Mean Vector μ")
st.write(mu)

st.subheader("Covariance Matrix Σ")
st.write(Sigma)

st.subheader("Inverse Covariance Σ⁻¹")
st.write(Sigma_inv)

# =====================================================
# MAHALANOBIS DISTANCE
# =====================================================
st.divider()
st.header("📐 Mahalanobis Distance Computation")

dM = mahalanobis_distance(y, mu, Sigma_inv)

st.write("Mahalanobis distance d_M(y, μ) =")
st.code(round(dM, 4))

# =====================================================
# PART (c) — OUTLIER DECISION
# =====================================================
st.divider()
st.header("🚨 (c) Outlier Detection")

threshold = 3.0

st.write("Decision rule:")
st.write("If distance > 3 → likely outlier")

st.write("Computed distance:", round(dM, 4))

if dM > threshold:
    st.error("❌ The point is likely an OUTLIER.")
else:
    st.success("✅ The point belongs to the class (NOT an outlier).")

# =====================================================
# OPTIONAL: EUCLIDEAN COMPARISON
# =====================================================
st.divider()
st.header("📏 Euclidean Distance Comparison")

euclid = np.linalg.norm(y - mu)
st.write("Euclidean distance from mean:")
st.code(round(euclid, 4))

st.markdown("""
Notice:
- Euclidean distance ignores data spread and correlation.
- Mahalanobis distance normalizes by covariance.
""")

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Visualization")

fig, ax = plt.subplots(figsize=(7, 6))

# Plot dataset
ax.scatter(C[:,0], C[:,1], s=80, label="Class samples")

# Plot mean
ax.scatter(mu[0], mu[1], marker="*", s=250, label="Mean μ")

# Plot test point
ax.scatter(y[0], y[1], marker="x", s=150, label="Test point y")

# Draw line from mean to test point
ax.plot([mu[0], y[0]], [mu[1], y[1]], "--", alpha=0.6)

ax.set_xlabel("Weight")
ax.set_ylabel("Height")
ax.set_title("Mahalanobis Outlier Detection")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# =====================================================
# INTERACTIVE EXPERIMENT
# =====================================================
st.divider()
st.header("🎛 Interactive Experiment")

y1 = st.slider("Test point weight", 0.0, 20.0, float(y[0]), 0.5)
y2 = st.slider("Test point height", 0.0, 100.0, float(y[1]), 1.0)

y_new = np.array([y1, y2])
dM_new = mahalanobis_distance(y_new, mu, Sigma_inv)

st.write("Updated Mahalanobis distance:", round(dM_new, 4))

if dM_new > threshold:
    st.warning("⚠️ This point is an OUTLIER.")
else:
    st.success("✅ This point belongs to the class.")
