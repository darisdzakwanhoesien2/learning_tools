import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Radial Lens Distortion Solver", layout="wide")
st.title("📸 Radial Lens Distortion — Step-by-Step Solver")

# =====================================================
# MODEL EQUATIONS
# =====================================================
st.header("📘 Radial Distortion Model")

st.latex(r"x_d = x_n + (x_n - x_{n0}) (k_1 r^2)")
st.latex(r"y_d = y_n + (y_n - y_{n0}) (k_1 r^2)")
st.latex(r"r = \sqrt{(x_n - x_{n0})^2 + (y_n - y_{n0})^2}")

st.markdown("""
**Assumptions**
- Only **k₁** is non-zero  
- Higher-order coefficients ignored  
""")

# =====================================================
# GIVEN VALUES
# =====================================================
st.header("📥 Given Values")

xn = np.array([100.0, 100.0])
x0 = np.array([300.0, 300.0])
xd = np.array([50.0, 50.0])

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Normalized Point**")
    st.latex(r"x_n = \begin{bmatrix}100 \\ 100\end{bmatrix}")

with c2:
    st.markdown("**Distortion Center**")
    st.latex(r"x_{n0} = \begin{bmatrix}300 \\ 300\end{bmatrix}")

with c3:
    st.markdown("**Distorted Point**")
    st.latex(r"x_d = \begin{bmatrix}50 \\ 50\end{bmatrix}")

# =====================================================
# STEP 1 — COMPUTE r
# =====================================================
st.divider()
st.header("🧮 Step 1 — Compute Radius r")

delta = xn - x0
r = float(np.linalg.norm(delta))

st.latex(r"r = \sqrt{(x_n - x_{n0})^2 + (y_n - y_{n0})^2}")

latex_r = r"""
r = \sqrt{{({dx})^2 + ({dy})^2}} = {rval:.4f}
""".format(
    dx=f"{delta[0]:.1f}",
    dy=f"{delta[1]:.1f}",
    rval=r
)

st.latex(latex_r)

# =====================================================
# STEP 2 — SOLVE FOR k1
# =====================================================
st.divider()
st.header("🧮 Step 2 — Solve for $k_1$")

st.latex(r"x_d = x_n + (x_n - x_{n0}) k_1 r^2")
st.latex(r"k_1 = \frac{x_d - x_n}{(x_n - x_{n0}) r^2}")

numerator = xd[0] - xn[0]
denominator = (xn[0] - x0[0]) * (r ** 2)
k1 = numerator / denominator

latex_k1 = (
    r"k_1 = "
    r"\frac{("
    + f"{xd[0]:.1f}"
    + r" - "
    + f"{xn[0]:.1f}"
    + r")}"
    r"{("
    + f"{xn[0]:.1f}"
    + r" - "
    + f"{x0[0]:.1f}"
    + r") \cdot "
    + f"{r**2:.2f}"
    + r")}"
    + r" = "
    + f"{k1:.6e}"
)

st.latex(latex_k1)

st.latex(latex_k1)
st.success(f"✅ Estimated k₁ = {k1:.6e}")

# =====================================================
# STEP 3 — VERIFICATION
# =====================================================
st.divider()
st.header("🔎 Step 3 — Verification")

xd_hat = xn + (xn - x0) * k1 * (r ** 2)

st.latex(r"\hat{x}_d = x_n + (x_n - x_{n0}) k_1 r^2")

st.write("Predicted distorted point:")
st.write(xd_hat)

st.write("Measured distorted point:")
st.write(xd)

error = np.linalg.norm(xd_hat - xd)
st.latex(r"\text{Reprojection error} = " + f"{error:.6e}")

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Visualization")

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(xn[0], xn[1], s=120, label="Original $x_n$")
ax.scatter(xd[0], xd[1], s=120, label="Measured $x_d$")
ax.scatter(xd_hat[0], xd_hat[1], s=120, marker="x", label="Predicted $\hat{x}_d$")
ax.scatter(x0[0], x0[1], s=120, label="Center $x_{n0}$")

ax.plot([x0[0], xn[0]], [x0[1], xn[1]], "--", alpha=0.5)
ax.plot([x0[0], xd[0]], [x0[1], xd[1]], "--", alpha=0.5)

ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()
ax.set_title("Radial Distortion Geometry")

st.pyplot(fig)

# =====================================================
# INTERACTIVE EXPERIMENT
# =====================================================
st.divider()
st.header("🎛 Interactive Experiment")

xn_x = st.slider("x_n.x", 0, 600, int(xn[0]))
xn_y = st.slider("x_n.y", 0, 600, int(xn[1]))
xd_x = st.slider("x_d.x", 0, 600, int(xd[0]))
xd_y = st.slider("x_d.y", 0, 600, int(xd[1]))

xn_new = np.array([xn_x, xn_y], dtype=float)
xd_new = np.array([xd_x, xd_y], dtype=float)

delta_new = xn_new - x0
r_new = np.linalg.norm(delta_new)

if abs(xn_new[0] - x0[0]) > 1e-6 and r_new > 1e-6:
    k1_new = (xd_new[0] - xn_new[0]) / ((xn_new[0] - x0[0]) * r_new**2)
    st.success(f"Updated k₁ = {k1_new:.6e}")
else:
    st.warning("Degenerate configuration — cannot compute k₁.")
