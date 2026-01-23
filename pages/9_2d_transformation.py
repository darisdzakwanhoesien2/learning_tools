import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="2D Transformation Recovery", layout="wide")
st.title("📐 2D Transformation Parameter Recovery")

# =====================================================
# INPUT DATA (FROM PROBLEM)
# =====================================================

p = np.array([
    [0,  2,  4,  6,  8, 10, 10,  8,  6,  4,  2, 0],
    [1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2, 2]
], dtype=float)

p_prime = np.array([
    [2,  4,  6,  8, 10, 12, 14, 12, 10,  8,  6, 4],
    [1.00, 1.04, 1.16, 1.36, 1.64, 2.00,
     3.00, 2.64, 2.36, 2.16, 2.04, 2.00]
], dtype=float)

x, y = p
xp, yp = p_prime
N = len(x)

# =====================================================
# MODEL DESCRIPTION
# =====================================================

st.header("📘 Transformation Model")

st.latex(r"""
x' = a x + b
""")

st.latex(r"""
y' = c y + d x^2
""")

st.markdown("""
Unknown parameters:

\[
\theta = [a, b, c, d]^T
\]
""")

# =====================================================
# BUILD LINEAR SYSTEM
# =====================================================

st.header("🧮 Step 1 — Build Linear System")

"""
For each point i:

x'_i = a x_i + b  
y'_i = c y_i + d x_i^2

This can be written as:

A θ = b
"""

A = []
b_vec = []

for i in range(N):
    xi, yi = x[i], y[i]
    xpi, ypi = xp[i], yp[i]

    # Equation for x'
    A.append([xi, 1, 0, 0])
    b_vec.append(xpi)

    # Equation for y'
    A.append([0, 0, yi, xi**2])
    b_vec.append(ypi)

A = np.array(A)
b_vec = np.array(b_vec)

st.write("Matrix A:")
st.write(A)

st.write("Vector b:")
st.write(b_vec)

# =====================================================
# SOLVE SYSTEM
# =====================================================

st.header("🧠 Step 2 — Solve Parameters")

theta_exact = np.linalg.lstsq(A, b_vec, rcond=None)[0]
a, b, c, d = theta_exact

st.success("Recovered parameters:")
st.latex(fr"a = {a:.4f}, \quad b = {b:.4f}")
st.latex(fr"c = {c:.4f}, \quad d = {d:.4f}")

# =====================================================
# PREDICTION
# =====================================================

xp_hat = a * x + b
yp_hat = c * y + d * x**2

# =====================================================
# VISUALIZATION
# =====================================================

st.header("📊 Step 3 — Visualization")

fig, ax = plt.subplots(1, 2, figsize=(12,5))

# Original
ax[0].scatter(x, y, c="blue")
ax[0].set_title("Original Points p")
ax[0].set_aspect("equal")
ax[0].grid(True)

# Transformed
ax[1].scatter(xp, yp, c="red", label="Measured p'")
ax[1].scatter(xp_hat, yp_hat, c="green", marker="x", label="Model Fit")
ax[1].set_title("Recovered Transformation")
ax[1].legend()
ax[1].set_aspect("equal")
ax[1].grid(True)

st.pyplot(fig)

# =====================================================
# RESIDUAL ANALYSIS
# =====================================================

st.header("📉 Step 4 — Residual Error")

residuals = np.sqrt((xp_hat - xp)**2 + (yp_hat - yp)**2)

st.write("Residuals per point:")
st.write(residuals)

st.latex(fr"\text{{Mean residual}} = {np.mean(residuals):.6f}")

# =====================================================
# NOISE DISCUSSION
# =====================================================

st.divider()
st.header("📌 What if there was noise?")

st.markdown("""
If measurements contain noise:

### ✅ Correct approach:
Use **least squares estimation**:

\[
\hat{\theta} = \arg\min_\theta \|A\theta - b\|^2
\]

Advantages:
- Robust to noise
- Uses all measurements
- Minimizes global error

### ⚠️ If noise is high:
- Regularization may be required
- Outlier rejection (RANSAC)
- Weighted least squares

In this app, we already use least squares internally.
""")
