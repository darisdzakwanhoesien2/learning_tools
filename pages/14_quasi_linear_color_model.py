import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Quasi-Linear Color Model", layout="wide")
st.title("🎨 Quasi-Linear Color Model — Polynomial Regression")

# =====================================================
# MODEL FORMULATION
# =====================================================

st.header("📘 Transformation Model")

st.latex(r"""
r = c_1 x + c_2 x^2 + c_3 y + c_4 z
""")

st.latex(r"""
g = c_5 x + c_6 y + c_7 y^3 + c_8 z
""")

st.latex(r"""
b = c_9 x + c_{10} y + c_{11} z + c_{12} z^2
""")

st.latex(r"""
\mathbf{c} =
[c_1, \ldots, c_{12}]^T
""")

# =====================================================
# INPUT DATA
# =====================================================

st.header("📥 Input Data")

# XYZ values from previous exercise
X = np.array([
    [ 9, 80, 19, 14, 55, 62, 40, 13, 42],
    [40, 43, 27, 87, 15, 35,  8, 19,  5],
    [26, 91, 15, 58, 85, 51, 24, 24, 90]
], dtype=float)

# RGB values for quasi-linear model
R = np.array([
    [   44,  2726,   161,   103,  1298,  1641,   687,    80,   765],
    [37169, 46238, 11461, 382032, 2030, 24964,   344,  4010,   123],
    [   52,   277,    31,   151,   235,   122,    44,    42,   249]
], dtype=float)

st.markdown("### XYZ samples")
st.write(X)

st.markdown("### RGB measurements")
st.write(R)

N = X.shape[1]

# =====================================================
# BUILD DESIGN MATRIX
# =====================================================

st.header("🧮 Step 1 — Build Design Matrix")

st.markdown("""
For each sample:

\[
\phi(x,y,z) =
[x,\; x^2,\; y,\; z,\;
 x,\; y,\; y^3,\; z,\;
 x,\; y,\; z,\; z^2]
\]
""")

def build_design_matrix(X):
    Phi = []
    for i in range(X.shape[1]):
        x, y, z = X[:, i]
        Phi.append([
            x, x**2, y, z,
            x, y, y**3, z,
            x, y, z, z**2
        ])
    return np.array(Phi)

Phi = build_design_matrix(X)

st.write("Design matrix Φ:")
st.write(Phi)

# =====================================================
# STACK OUTPUT VECTOR
# =====================================================

st.header("🧮 Step 2 — Stack Output Vector")

y = np.hstack([
    R[0, :],
    R[1, :],
    R[2, :]
])

# Build block diagonal Φ
Phi_big = np.zeros((3*N, 12))

for i in range(N):
    Phi_big[3*i + 0, 0:4]   = Phi[i, 0:4]
    Phi_big[3*i + 1, 4:8]   = Phi[i, 4:8]
    Phi_big[3*i + 2, 8:12]  = Phi[i, 8:12]

st.write("Expanded regression matrix:")
st.write(Phi_big)

# =====================================================
# SOLVE LEAST SQUARES
# =====================================================

st.header("🧠 Step 3 — Solve for Coefficients")

c, residuals, rank, s = np.linalg.lstsq(Phi_big, y, rcond=None)

st.success("Estimated coefficient vector c:")
st.write(c)

st.subheader("Coefficient Blocks")

st.write("r-coefficients:", c[0:4])
st.write("g-coefficients:", c[4:8])
st.write("b-coefficients:", c[8:12])

# =====================================================
# PREDICTION + ERROR
# =====================================================

st.header("📉 Step 4 — Prediction Accuracy")

def predict_rgb(X, c):
    C = []
    for i in range(X.shape[1]):
        x, y, z = X[:, i]

        r = c[0]*x + c[1]*x**2 + c[2]*y + c[3]*z
        g = c[4]*x + c[5]*y + c[6]*y**3 + c[7]*z
        b = c[8]*x + c[9]*y + c[10]*z + c[11]*z**2

        C.append([r,g,b])

    return np.array(C).T

R_hat = predict_rgb(X, c)
error = np.linalg.norm(R_hat - R)

st.latex(fr"\|R_{{pred}} - R\|_F = {error:.3f}")

# =====================================================
# VISUALIZATION
# =====================================================

st.header("📊 Prediction Quality")

fig, ax = plt.subplots(1,3, figsize=(15,4))

labels = ["R", "G", "B"]

for i in range(3):
    ax[i].scatter(R[i,:], R_hat[i,:])
    ax[i].plot([R.min(), R.max()], [R.min(), R.max()], 'k--')
    ax[i].set_title(f"{labels[i]} channel")
    ax[i].set_xlabel("True")
    ax[i].set_ylabel("Predicted")
    ax[i].grid(True)

st.pyplot(fig)

# =====================================================
# DISCUSSION
# =====================================================

st.divider()
st.header("🧠 Discussion")

st.markdown("""
### Why polynomial models?

- Capture nonlinear sensor behavior.
- Correct channel coupling.
- Improve color accuracy.

### Risks:

- Overfitting if data is limited.
- Numerical conditioning problems.
- Sensitive to noise.

### Best practice:

- Normalize inputs.
- Cross-validate.
- Regularize if needed.
""")

st.caption("🚀 Quasi-linear polynomial color calibration pipeline.")
