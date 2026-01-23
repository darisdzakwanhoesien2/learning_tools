import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Linear Color Space Estimation", layout="wide")
st.title("🎨 Linear Color Spaces — XYZ → RGB Estimation")

# =====================================================
# PROBLEM FORMULATION
# =====================================================

st.header("📘 Transformation Model")

st.latex(r"""
r = c_1 x + c_2 y + c_3 z
""")

st.latex(r"""
g = c_4 x + c_5 y + c_6 z
""")

st.latex(r"""
b = c_7 x + c_8 y + c_9 z
""")

st.latex(r"""
\mathbf{c} =
[c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9]^T
""")

# =====================================================
# GIVEN DATA
# =====================================================

st.header("📥 Given Data")

X = np.array([
    [ 9, 80, 19, 14, 55, 62, 40, 13, 42],
    [40, 43, 27, 87, 15, 35,  8, 19,  5],
    [26, 91, 15, 58, 85, 51, 24, 24, 90]
], dtype=float)

R = np.array([
    [17, 58, 11, 38, 51, 35, 17, 15, 52],
    [48,116, 42,101, 69, 88, 42, 31, 50],
    [25, 69, 24, 51, 41, 54, 28, 17, 29]
], dtype=float)

st.markdown("### XYZ measurements (X)")
st.write(X)

st.markdown("### Corresponding RGB measurements (R)")
st.write(R)

N = X.shape[1]

# =====================================================
# BUILD LINEAR SYSTEM
# =====================================================

st.header("🧮 Build Linear System")

st.markdown("""
Each color pair produces three equations:

\[
\begin{bmatrix}
x & y & z & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & x & y & z & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & x & y & z
\end{bmatrix}
\mathbf{c}
=
\begin{bmatrix}
r \\ g \\ b
\end{bmatrix}
\]
""")

def build_system(X, R, num_samples):
    A = []
    b = []
    for i in range(num_samples):
        x, y, z = X[:, i]
        r, g, b_val = R[:, i]

        A.append([x,y,z, 0,0,0, 0,0,0])
        A.append([0,0,0, x,y,z, 0,0,0])
        A.append([0,0,0, 0,0,0, x,y,z])

        b.append(r)
        b.append(g)
        b.append(b_val)

    return np.array(A), np.array(b)

# =====================================================
# PART (a) MINIMUM DATA SOLUTION
# =====================================================

st.divider()
st.header("🟢 (a) Minimum Number of Color Pairs")

st.markdown("""
We need at least **3 color pairs**:

- Each color gives 3 equations
- Total unknowns = 9
- Minimum equations = 9 → 3 samples
""")

A_min, b_min = build_system(X, R, num_samples=3)

st.write("Matrix A (minimum system):")
st.write(A_min)

st.write("Vector b:")
st.write(b_min)

c_min = np.linalg.solve(A_min, b_min)

st.success("Estimated coefficients (minimum data):")
st.write(c_min.reshape(3,3))

# =====================================================
# PART (b) LEAST SQUARES SOLUTION
# =====================================================

st.divider()
st.header("🟡 (b) Using All Color Pairs (Least Squares)")

A_all, b_all = build_system(X, R, num_samples=N)

st.write("Matrix A (full system):")
st.write(A_all)

st.write("Vector b:")
st.write(b_all)

c_ls, residuals, rank, s = np.linalg.lstsq(A_all, b_all, rcond=None)

st.success("Estimated coefficients (least squares):")
st.write(c_ls.reshape(3,3))

# =====================================================
# ERROR ANALYSIS
# =====================================================

st.divider()
st.header("📉 Reconstruction Error Comparison")

def predict(X, c):
    C = c.reshape(3,3)
    return C @ X

R_min_pred = predict(X, c_min)
R_ls_pred  = predict(X, c_ls)

err_min = np.linalg.norm(R_min_pred - R)
err_ls  = np.linalg.norm(R_ls_pred  - R)

st.latex(fr"\text{{Error (minimum)}} = {err_min:.3f}")
st.latex(fr"\text{{Error (least squares)}} = {err_ls:.3f}")

# =====================================================
# VISUALIZATION
# =====================================================

st.header("📊 Prediction Visualization")

fig, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].scatter(R.flatten(), R_min_pred.flatten())
ax[0].set_title("Minimum Data Fit")
ax[0].set_xlabel("True RGB")
ax[0].set_ylabel("Predicted RGB")
ax[0].grid(True)

ax[1].scatter(R.flatten(), R_ls_pred.flatten())
ax[1].set_title("Least Squares Fit")
ax[1].set_xlabel("True RGB")
ax[1].set_ylabel("Predicted RGB")
ax[1].grid(True)

st.pyplot(fig)

# =====================================================
# DISCUSSION
# =====================================================

st.divider()
st.header("🧠 Discussion")

st.markdown("""
### Why least squares is better:
- Measurements contain **quantization noise**.
- Minimum solution fits exactly only 3 samples.
- Least squares minimizes global error across all samples.
- More robust and stable.

### In practice:
Always use more measurements than unknowns when noise exists.
""")

st.caption("🚀 Fully reproducible linear color calibration solver.")
