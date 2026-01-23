import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Spatial Moments & Ellipse Estimation", layout="wide")
st.title("🟠 Spatial Moments — Ellipse Coefficient Estimation")

# =====================================================
# INPUT BINARY IMAGE (FROM PROBLEM)
# =====================================================

B = np.array([
 [0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,1,0,0],
 [0,0,0,0,0,1,1,1,1,1,0,0,0],
 [0,0,0,0,1,1,1,1,1,0,0,0,0],
 [0,0,0,1,1,1,1,1,0,0,0,0,0],
 [0,0,1,1,1,1,0,0,0,0,0,0,0],
 [0,1,1,1,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.uint8)

H, W = B.shape

st.header("📥 Binary Region Image")
st.dataframe(B)

# =====================================================
# STEP 1 — COORDINATE SYSTEM
# =====================================================

st.divider()
st.header("📐 Step 1 — Coordinate System")

st.markdown("""
We assume the origin is at the image center.

Pixel coordinates:

\[
x = j - \frac{W-1}{2}, \quad
y = \frac{H-1}{2} - i
\]
""")

ys, xs = np.where(B == 1)

x = xs - (W-1)/2
y = (H-1)/2 - ys

points = np.vstack([x,y]).T

st.write("Foreground coordinates (x, y):")
st.dataframe(points)

# =====================================================
# STEP 2 — SPATIAL MOMENTS
# =====================================================

st.divider()
st.header("🧮 Step 2 — Spatial Moments")

m00 = len(points)
m10 = np.sum(x)
m01 = np.sum(y)

x_bar = m10 / m00
y_bar = m01 / m00

st.latex(fr"""
m_{{00}} = {m00}, \quad
m_{{10}} = {m10:.2f}, \quad
m_{{01}} = {m01:.2f}
""")

st.latex(fr"""
\bar x = {x_bar:.3f}, \quad \bar y = {y_bar:.3f}
""")

# Central moments
mu20 = np.sum((x - x_bar)**2)
mu02 = np.sum((y - y_bar)**2)
mu11 = np.sum((x - x_bar)*(y - y_bar))

st.latex(fr"""
\mu_{{20}} = {mu20:.2f}, \quad
\mu_{{02}} = {mu02:.2f}, \quad
\mu_{{11}} = {mu11:.2f}
""")

# =====================================================
# STEP 3 — COVARIANCE MATRIX
# =====================================================

st.divider()
st.header("📊 Step 3 — Covariance Matrix")

Sigma = np.array([
    [mu20/m00, mu11/m00],
    [mu11/m00, mu02/m00]
])

st.latex(r"""
\Sigma =
\begin{bmatrix}
\sigma_{xx} & \sigma_{xy} \\
\sigma_{xy} & \sigma_{yy}
\end{bmatrix}
""")

st.write(Sigma)

# =====================================================
# STEP 4 — ELLIPSE QUADRATIC FORM
# =====================================================

st.divider()
st.header("🔍 Step 4 — Ellipse Equation")

st.markdown("""
Ellipse defined as:

\[
[x \; y]
\Sigma^{-1}
\begin{bmatrix} x \\ y \end{bmatrix} = 1
\]

Expands to:

\[
d x^2 + 2 e x y + f y^2 = 1
\]
""")

Sigma_inv = np.linalg.inv(Sigma)

d = Sigma_inv[0,0]
e = Sigma_inv[0,1]
f = Sigma_inv[1,1]

st.latex(fr"""
d = {d:.4f}, \quad
e = {e:.4f}, \quad
f = {f:.4f}
""")

# =====================================================
# STEP 5 — VISUALIZATION
# =====================================================

st.divider()
st.header("📈 Step 5 — Visualization")

theta = np.linspace(0, 2*np.pi, 400)
eigvals, eigvecs = np.linalg.eigh(Sigma)

a = np.sqrt(eigvals[0])
b = np.sqrt(eigvals[1])

ellipse = np.array([
    a * np.cos(theta),
    b * np.sin(theta)
])

ellipse = eigvecs @ ellipse

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(x, y, label="Region Pixels")
ax.plot(ellipse[0], ellipse[1], 'r', linewidth=2, label="Estimated Ellipse")
ax.set_aspect("equal")
ax.grid(True)
ax.legend()
ax.set_title("Ellipse Fit from Spatial Moments")

st.pyplot(fig)

# =====================================================
# INTERPRETATION
# =====================================================

st.divider()
st.header("🧠 Interpretation")

st.markdown("""
- Moments capture second-order geometry.
- Covariance describes spread of region.
- Inverse covariance defines ellipse metric.
- Coefficients (d,e,f) define the quadratic form.

This method assumes:
- Uniform density inside region.
- Elliptical approximation.

Commonly used in shape analysis and object detection.
""")

st.caption("🚀 Spatial moments ellipse estimation complete.")
