import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Gaussian Decision Boundary (2D)",
    layout="wide"
)
st.title("📈 Decision Boundary for Two-Dimensional Gaussian Data")

# =====================================================
# TRAINING DATA
# =====================================================
# ω1 samples: (-1,6), (0,8), (0,4), (1,6)
W1 = np.array([
    [-1, 6],
    [ 0, 8],
    [ 0, 4],
    [ 1, 6]
], dtype=float)

# ω2 samples: (-2,-2), (0,0), (0,-4), (2,-2)
W2 = np.array([
    [-2, -2],
    [ 0,  0],
    [ 0, -4],
    [ 2, -2]
], dtype=float)

# =====================================================
# DISPLAY DATA
# =====================================================
st.header("📥 Training Samples")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Class ω₁")
    st.write(W1)

with c2:
    st.subheader("Class ω₂")
    st.write(W2)

# =====================================================
# HELPERS
# =====================================================
def sample_mean(X):
    return X.mean(axis=0)

def sample_covariance(X):
    """
    Σ = (1/N) Σ (x_n - μ)(x_n - μ)^T
    """
    mu = sample_mean(X)
    N = X.shape[0]
    S = np.zeros((2, 2))
    for x in X:
        d = (x - mu).reshape(2, 1)
        S += d @ d.T
    return S / N

def gaussian_discriminant(x, mu, Sigma, prior=0.5):
    """
    g(x) = -1/2 (x-μ)^T Σ^{-1} (x-μ)
           -1/2 ln |Σ| + ln P(ω)
    """
    invS = np.linalg.inv(Sigma)
    detS = np.linalg.det(Sigma)
    d = x - mu
    return float(
        -0.5 * d.T @ invS @ d
        -0.5 * np.log(detS)
        + np.log(prior)
    )

# =====================================================
# PART (a) — MEAN AND COVARIANCE
# =====================================================
st.divider()
st.header("🧮 (a) Estimate Mean and Covariance")

mu1 = sample_mean(W1)
mu2 = sample_mean(W2)

Sigma1 = sample_covariance(W1)
Sigma2 = sample_covariance(W2)

st.subheader("Class ω₁ Parameters")
st.write("Mean μ₁:", mu1)
st.write("Covariance Σ₁:")
st.write(Sigma1)

st.subheader("Class ω₂ Parameters")
st.write("Mean μ₂:", mu2)
st.write("Covariance Σ₂:")
st.write(Sigma2)

# =====================================================
# PART (b) — DECISION BOUNDARY
# =====================================================
st.divider()
st.header("📐 (b) Bayes Decision Boundary")

st.markdown("""
We assume **equal priors**:

P(ω₁) = P(ω₂)

The decision boundary is defined by:

g₁(x) = g₂(x)

which forms a **second-degree curve (quadratic)** in the plane.
""")

# =====================================================
# GRID EVALUATION (FAST VECTORIZED)
# =====================================================
xmin, xmax = -5.0, 5.0
ymin, ymax = -6.0, 10.0

xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 300),
    np.linspace(ymin, ymax, 300)
)

grid = np.stack([xx, yy], axis=-1)

def batch_discriminant(grid, mu, Sigma):
    invS = np.linalg.inv(Sigma)
    detS = np.linalg.det(Sigma)
    d = grid - mu
    quad = np.einsum("...i,ij,...j->...", d, invS, d)
    return -0.5 * quad - 0.5 * np.log(detS)

Z = batch_discriminant(grid, mu1, Sigma1) - batch_discriminant(grid, mu2, Sigma2)

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Decision Boundary Visualization")

fig, ax = plt.subplots(figsize=(8, 7))

# Plot samples
ax.scatter(W1[:,0], W1[:,1], s=120, label="Class ω₁")
ax.scatter(W2[:,0], W2[:,1], s=120, label="Class ω₂")

# Plot means
ax.scatter(mu1[0], mu1[1], marker="*", s=250, label="μ₁")
ax.scatter(mu2[0], mu2[1], marker="*", s=250, label="μ₂")

# Decision boundary
ax.contour(xx, yy, Z, levels=[0], linewidths=2)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Quadratic Bayes Decision Boundary")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# =====================================================
# INTERACTIVE TEST POINT
# =====================================================
st.divider()
st.header("🎯 Test a Point")

tx = st.slider("x coordinate", xmin, xmax, 0.0, 0.1)
ty = st.slider("y coordinate", ymin, ymax, 0.0, 0.1)

test_point = np.array([tx, ty])

g1 = gaussian_discriminant(test_point, mu1, Sigma1)
g2 = gaussian_discriminant(test_point, mu2, Sigma2)

st.write("g₁(x) =", round(g1, 4))
st.write("g₂(x) =", round(g2, 4))

if g1 > g2:
    st.success("✅ Classified as class ω₁")
else:
    st.success("✅ Classified as class ω₂")

st.caption("🚀 Bayes decision boundary for 2D Gaussian classes.")
