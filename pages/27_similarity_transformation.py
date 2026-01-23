import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Similarity Transformation Solver",
    layout="wide"
)
st.title("🔁 Similarity Transformation from Two Point Correspondences")

# =====================================================
# GIVEN POINT CORRESPONDENCES
# =====================================================
# p -> p'
P = np.array([
    [0.5, 0.0],     # p1
    [0.0, 0.5]      # p2
], dtype=float)

Pp = np.array([
    [0.0,  0.0],    # p1'
    [-1.0, -1.0]    # p2'
], dtype=float)

# =====================================================
# DISPLAY DATA
# =====================================================
st.header("📥 Given Correspondences")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Original Points P")
    st.write(P)

with c2:
    st.subheader("Transformed Points P'")
    st.write(Pp)

# =====================================================
# PART (a) — METHOD DESCRIPTION
# =====================================================
st.divider()
st.header("🧠 (a) How to Estimate a Similarity Transformation")

st.markdown("""
A 2D similarity transformation has the form:

x' = s R x + t

where:
- s is a scale factor
- R is a 2×2 rotation matrix
- t is a translation vector

Steps to solve:
1. Subtract centroids of both point sets.
2. Estimate rotation and scale using least squares / SVD.
3. Recover translation from centroid alignment.
4. Build the final transformation matrix.

With **two point correspondences**, the system is exactly determined.
""")

# =====================================================
# PART (b) — SOLVE PARAMETERS
# =====================================================
st.divider()
st.header("🧮 (b) Solve Transformation Parameters")

# Compute centroids
mu_P  = P.mean(axis=0)
mu_Pp = Pp.mean(axis=0)

X = P  - mu_P
Y = Pp - mu_Pp

# Cross-covariance
H = X.T @ Y

# SVD
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T

# Ensure proper rotation (det = +1)
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T

# Scale
var_X = np.sum(X**2)
s = np.trace(np.diag(S)) / var_X

# Translation
t = mu_Pp - s * R @ mu_P

# =====================================================
# DISPLAY RESULTS
# =====================================================
st.subheader("Estimated Parameters")

st.write("Scale s:")
st.code(round(float(s), 4))

st.write("Rotation R:")
st.write(R)

st.write("Translation t:")
st.write(t)

# =====================================================
# BUILD TRANSFORMATION MATRIX
# =====================================================
T = np.eye(3)
T[:2, :2] = s * R
T[:2, 2]  = t

st.subheader("Homogeneous Transformation Matrix")
st.write(T)

# =====================================================
# VERIFICATION
# =====================================================
st.divider()
st.header("✅ Verification")

def apply_transform(P, T):
    P_h = np.hstack([P, np.ones((P.shape[0], 1))])
    Pp_hat = (T @ P_h.T).T
    return Pp_hat[:, :2]

Pp_est = apply_transform(P, T)

st.write("Predicted transformed points:")
st.write(Pp_est)

st.write("Ground truth transformed points:")
st.write(Pp)

error = np.linalg.norm(Pp_est - Pp)
st.write("Reprojection error:", round(error, 6))

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Visualization")

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(P[:,0],  P[:,1],  s=120, label="Original P")
ax.scatter(Pp[:,0], Pp[:,1], s=120, label="Target P'")
ax.scatter(Pp_est[:,0], Pp_est[:,1], s=120, marker="x", label="Estimated P'")

for i in range(len(P)):
    ax.plot(
        [P[i,0], Pp_est[i,0]],
        [P[i,1], Pp_est[i,1]],
        "--",
        alpha=0.5
    )

ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.set_title("Similarity Transformation Mapping")
ax.legend()

st.pyplot(fig)

# =====================================================
# INTERACTIVE TEST
# =====================================================
st.divider()
st.header("🎛 Test a New Point")

tx = st.slider("x", -2.0, 2.0, 0.2, 0.05)
ty = st.slider("y", -2.0, 2.0, 0.2, 0.05)

test_point = np.array([[tx, ty]])
mapped = apply_transform(test_point, T)

st.write("Input point:")
st.write(test_point)

st.write("Mapped point:")
st.write(mapped)

st.caption("🚀 Similarity transformation estimation from two point correspondences.")
