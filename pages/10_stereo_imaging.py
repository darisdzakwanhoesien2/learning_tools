import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Stereo Imaging — Step by Step", layout="wide")
st.title("📷 Stereo Imaging — Step-by-Step Solution")

# =====================================================
# GIVEN DATA
# =====================================================
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]], dtype=float)

C2 = np.array([[1, 0, 0, 1],
               [0, 0, -1, 1],
               [0, 1, 0, 1]], dtype=float)

p  = np.array([-0.5, 0.5, 1.0], dtype=float)
p2 = np.array([0.0, -0.5, 1.0], dtype=float)

E = np.array([[0,  1,  1],
              [1, -1,  0],
              [-1, 0, -1]], dtype=float)

# =====================================================
# DISPLAY GIVEN VALUES
# =====================================================
st.header("📥 Given Data")

st.latex(r"C = \begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\end{bmatrix}")
st.latex(r"C' = \begin{bmatrix}1&0&0&1\\0&0&-1&1\\0&1&0&1\end{bmatrix}")

st.latex(r"p = \begin{bmatrix}-\frac12\\\frac12\\1\end{bmatrix}")
st.latex(r"p' = \begin{bmatrix}0\\-\frac12\\1\end{bmatrix}")

st.latex(r"E = \begin{bmatrix}0&1&1\\1&-1&0\\-1&0&-1\end{bmatrix}")

# =====================================================
# TRIANGULATION
# =====================================================
st.divider()
st.header("🧮 (a) Triangulation")

st.latex(r"X = \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}")
st.latex(r"p \sim CX \quad,\quad p' \sim C'X")
st.latex(r"p \times (CX) = 0 \quad,\quad p' \times (C'X) = 0")
st.latex(r"[p]_\times (CX) = 0 \quad,\quad [p']_\times (C'X) = 0")

# =====================================================
# SKEW MATRIX
# =====================================================
def skew(v):
    return np.array([[0,     -v[2],  v[1]],
                     [v[2],   0,    -v[0]],
                     [-v[1],  v[0],  0]], dtype=float)

Px  = skew(p)
P2x = skew(p2)

st.subheader("Skew Matrices")
st.write("[p]×")
st.dataframe(Px)
st.write("[p′]×")
st.dataframe(P2x)

# =====================================================
# LINEAR SYSTEM
# =====================================================
A = np.vstack([Px @ C, P2x @ C2])

st.subheader("Linear System")
st.latex(r"A X = 0")
st.dataframe(A)

# =====================================================
# SOLVE
# =====================================================
_, _, Vt = np.linalg.svd(A)
Xh = Vt[-1]
Xh = Xh / Xh[-1]
X, Y, Z = Xh[:3]

st.success("Recovered 3D Point")
st.latex("X = " + f"{X:.4f}" + r"\quad Y = " + f"{Y:.4f}" + r"\quad Z = " + f"{Z:.4f}")

# =====================================================
# EPIPOLAR LINE
# =====================================================
st.divider()
st.header("📐 (b) Epipolar Line")

st.latex(r"\ell' = E p")
l = E @ p

latex_line = (
    r"\ell' = \begin{bmatrix}"
    + f"{l[0]:.3f}" + r"\\"
    + f"{l[1]:.3f}" + r"\\"
    + f"{l[2]:.3f}"
    + r"\end{bmatrix}"
)
st.latex(latex_line)

val = float(p2.T @ l)
st.latex(r"p'^T \ell' = " + f"{val:.6f}")

# =====================================================
# VISUALIZATION
# =====================================================
a, b, c = l
xs = np.linspace(-2, 2, 300)
ys = -(a * xs + c) / b

fig, ax = plt.subplots()
ax.plot(xs, ys, label="Epipolar line")
ax.scatter(p2[0], p2[1], label="p'")
ax.set_aspect("equal")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# =====================================================
# EPIPOLAR MEANING
# =====================================================
st.divider()
st.header("🎯 (c) Epipolar Constraint Usage")

st.markdown(
    "The epipolar constraint reduces correspondence search from 2D to 1D. "
    "Matching occurs only along the epipolar line, improving efficiency and robustness."
)
