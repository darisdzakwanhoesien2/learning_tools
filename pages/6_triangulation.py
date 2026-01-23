import streamlit as st
import numpy as np

st.set_page_config(page_title="1D Camera Triangulation", layout="wide")

st.title("📐 1D Camera Triangulation Explorer")

# =====================================================
# THEORY — MATHEMATICAL MODEL
# =====================================================

st.header("📘 Camera Model")

st.markdown("We use homogeneous coordinates:")

st.latex(r"m \propto P x")

st.latex(r"""
m =
\begin{bmatrix}
m \\ 1
\end{bmatrix},
\quad
x =
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix},
\quad
P \in \mathbb{R}^{2\times 3}
""")

st.markdown("For a camera")

st.latex(r"""
P =
\begin{bmatrix}
p_{11} & p_{12} & p_{13} \\
p_{21} & p_{22} & p_{23}
\end{bmatrix}
""")

st.markdown("The projection equation becomes:")

st.latex(r"""
m = \frac{p_{11}x + p_{12}y + p_{13}}
         {p_{21}x + p_{22}y + p_{23}}
""")

st.markdown("Multiply both sides:")

st.latex(r"""
m(p_{21}x + p_{22}y + p_{23})
= p_{11}x + p_{12}y + p_{13}
""")

st.markdown("Rearranging into linear form:")

st.latex(r"""
(p_{11} - m p_{21})x + (p_{12} - m p_{22})y
= m p_{23} - p_{13}
""")

st.markdown("Each camera provides one linear equation of the form:")

st.latex(r"""
a_1 x + a_2 y = b
""")

st.markdown("Stacking two cameras gives:")

st.latex(r"""
A
\begin{bmatrix}
x \\ y
\end{bmatrix}
=
b
""")

# =====================================================
# INPUTS
# =====================================================

st.sidebar.header("📥 Inputs")

P1 = st.sidebar.text_area(
    "Camera P1 (2×3 matrix, comma-separated rows)",
    "1,2,0\n2,1,0"
)

P2 = st.sidebar.text_area(
    "Camera P2 (2×3 matrix, comma-separated rows)",
    "1,2,3\n4,2,0"
)

m1 = st.sidebar.number_input("Measurement m1", value=1.25, step=0.01)
m2 = st.sidebar.number_input("Measurement m2", value=1.0, step=0.01)

# =====================================================
# UTILITIES
# =====================================================

def parse_matrix(text):
    rows = []
    for line in text.strip().split("\n"):
        rows.append([float(v) for v in line.split(",")])
    return np.array(rows)

def build_constraint(P, m):
    """
    Build a single row of A and scalar b from projection matrix P and measurement m.
    """
    p11, p12, p13 = P[0]
    p21, p22, p23 = P[1]

    a1 = p11 - m * p21
    a2 = p12 - m * p22
    b = m * p23 - p13

    return np.array([a1, a2]), b

# =====================================================
# COMPUTATION
# =====================================================

try:
    P1_mat = parse_matrix(P1)
    P2_mat = parse_matrix(P2)

    a1, b1 = build_constraint(P1_mat, m1)
    a2, b2 = build_constraint(P2_mat, m2)

    A = np.vstack([a1, a2])
    b = np.array([b1, b2])

    x_est = np.linalg.solve(A, b)

    success = True
except Exception as e:
    success = False
    st.error(f"❌ Error parsing or solving system: {e}")

# =====================================================
# STEP-BY-STEP OUTPUT
# =====================================================

st.header("🧮 Step-by-Step Computation")

if success:

    st.subheader("1️⃣ Camera Matrices")

    st.write("P1 =")
    st.write(P1_mat)

    st.write("P2 =")
    st.write(P2_mat)

    st.subheader("2️⃣ Measurements")

    st.latex(fr"m_1 = {m1}, \quad m_2 = {m2}")

    st.subheader("3️⃣ Linear Equations From Each Camera")

    # Camera 1 equation
    st.markdown("**Camera 1 equation:**")
    st.latex(
        fr"({P1_mat[0,0]} - {m1}\cdot{P1_mat[1,0]})x + "
        fr"({P1_mat[0,1]} - {m1}\cdot{P1_mat[1,1]})y"
        fr" = {m1}\cdot{P1_mat[1,2]} - {P1_mat[0,2]}"
    )

    st.latex(fr"{a1[0]:.3f}x + {a1[1]:.3f}y = {b1:.3f}")

    # Camera 2 equation
    st.markdown("**Camera 2 equation:**")
    st.latex(
        fr"({P2_mat[0,0]} - {m2}\cdot{P2_mat[1,0]})x + "
        fr"({P2_mat[0,1]} - {m2}\cdot{P2_mat[1,1]})y"
        fr" = {m2}\cdot{P2_mat[1,2]} - {P2_mat[0,2]}"
    )

    st.latex(fr"{a2[0]:.3f}x + {a2[1]:.3f}y = {b2:.3f}")

    st.subheader("4️⃣ Matrix Form")

    st.latex(r"""
    A =
    \begin{bmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22}
    \end{bmatrix},
    \quad
    b =
    \begin{bmatrix}
    b_1 \\ b_2
    \end{bmatrix}
    """)

    st.write("A =")
    st.write(A)

    st.write("b =")
    st.write(b)

    st.subheader("5️⃣ Solve Linear System")

    st.latex(r"""
    \begin{bmatrix}
    x \\ y
    \end{bmatrix}
    = A^{-1} b
    """)

    st.success(f"✅ Solution:")
    st.latex(fr"x = {x_est[0]:.4f}, \quad y = {x_est[1]:.4f}")

# =====================================================
# THEORY QUESTIONS (C & D)
# =====================================================

st.divider()

st.header("📌 (c) Multiple Images for One Point")

st.latex(r"""
\min_x \|Ax - b\|^2
""")

st.markdown("""
**Advantages:**
- Reduces noise sensitivity.
- Improves numerical stability.
- Produces statistically optimal estimates (least squares).
- Robust against outliers when extended.
""")

st.header("📌 (d) Joint Estimation of Many Points")

st.markdown("""
**Advantages:**
- Enforces global geometric consistency.
- Allows joint optimization of camera parameters.
- Reduces accumulated drift and bias.
- Foundation of bundle adjustment.

**Trade-off:** Increased computational cost and complexity.
""")

st.caption("🚀 Modify P1, P2, m1, and m2 to see every step update dynamically.")


# import streamlit as st
# import numpy as np

# st.set_page_config(page_title="1D Camera Triangulation", layout="wide")

# st.title("📐 1D Camera Triangulation Explorer")
# st.markdown("""
# This app demonstrates **triangulation with 1D cameras**.

# Model:

# m ∝ P x,  
# where
# - m = [m, 1]^T
# - x = [x, y, 1]^T
# - P is a 2×3 projection matrix

# From each camera we obtain one linear constraint on (x, y).

# For a camera P = [[p11, p12, p13], [p21, p22, p23]] and measurement m:

# m(p21 x + p22 y + p23) = (p11 x + p12 y + p13)

# Rearranged:

# (p11 − m p21) x + (p12 − m p22) y = m p23 − p13

# This yields a linear system:

# A [x y]^T = b
# """)

# # ==========================
# # INPUTS
# # ==========================

# st.sidebar.header("📥 Inputs")

# P1 = st.sidebar.text_area(
#     "Camera P1 (2x3 matrix, comma-separated rows)",
#     "1,2,0\n2,1,0"
# )

# P2 = st.sidebar.text_area(
#     "Camera P2 (2x3 matrix, comma-separated rows)",
#     "1,2,3\n4,2,0"
# )

# m1 = st.sidebar.number_input("Measurement m1", value=1.25, step=0.01)
# m2 = st.sidebar.number_input("Measurement m2", value=1.0, step=0.01)

# # ==========================
# # UTILITIES
# # ==========================

# def parse_matrix(text):
#     rows = []
#     for line in text.strip().split("\n"):
#         rows.append([float(v) for v in line.split(",")])
#     return np.array(rows)


# def build_constraint(P, m):
#     """
#     Build a single row of A and b from projection matrix P and measurement m.
#     """
#     p11, p12, p13 = P[0]
#     p21, p22, p23 = P[1]

#     a1 = p11 - m * p21
#     a2 = p12 - m * p22
#     b = m * p23 - p13

#     return np.array([a1, a2]), b

# # ==========================
# # COMPUTATION
# # ==========================

# try:
#     P1_mat = parse_matrix(P1)
#     P2_mat = parse_matrix(P2)

#     a1, b1 = build_constraint(P1_mat, m1)
#     a2, b2 = build_constraint(P2_mat, m2)

#     A = np.vstack([a1, a2])
#     b = np.array([b1, b2])

#     x_est = np.linalg.solve(A, b)

#     success = True
# except Exception as e:
#     success = False
#     st.error(f"❌ Error parsing or solving system: {e}")

# # ==========================
# # OUTPUT
# # ==========================

# st.header("📊 Linear System")

# if success:
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Matrix A")
#         st.write(A)

#     with col2:
#         st.subheader("Vector b")
#         st.write(b)

#     st.subheader("✅ Estimated Point (x, y)")
#     st.success(f"x = {x_est[0]:.4f},   y = {x_est[1]:.4f}")

# # ==========================
# # THEORY SECTIONS (C & D)
# # ==========================

# st.divider()

# st.header("📌 (c) Multiple Images for One Point")
# st.markdown("""
# **Yes, there is a strong advantage.**

# When a point is observed in many images:

# - Each image provides an independent constraint.
# - The system becomes over‑determined.
# - Noise and measurement errors are averaged out.
# - The estimate becomes more accurate and more stable.

# Instead of solving exactly A x = b, we solve a **least‑squares problem**:

# min ||A x − b||²

# This improves robustness and numerical stability.
# """)

# st.header("📌 (d) Joint Estimation of Many Points")
# st.markdown("""
# **Yes, joint estimation can be beneficial in structured problems.**

# Advantages:

# - Shared camera parameters can be optimized jointly.
# - Global consistency between points is enforced.
# - Correlated noise can be handled properly.
# - Enables techniques such as **bundle adjustment**.

# However:

# - Computational cost increases significantly.
# - For independent points with fixed cameras, solving each point separately is usually sufficient.
# """)

# st.caption("🚀 You can modify P1, P2, m1, and m2 in the sidebar to experiment interactively.")
