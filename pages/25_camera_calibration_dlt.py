import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Camera Calibration (DLT)",
    layout="wide"
)
st.title("📷 Camera Calibration — Projection Matrix Estimation (DLT)")

# =====================================================
# GIVEN DATA
# =====================================================
# 3D world points P_i = (X, Y, Z, 1)
P = np.array([
    [ 0,  0,  3, 1],
    [ 1,  0,  3, 1],
    [-1,  1,  2, 1],
    [-1,  0,  2, 1],
    [ 1,  1,  3, 1],
    [ 0,  1,  3, 1]
], dtype=float)

# Observed image points in second image p'_i = (x, y)
p_img = np.array([
    [ 1,  -2],
    [ 2,  -2],
    [ 0, -0.5],
    [ 0,  -1],
    [ 1,  -1],
    [ 2,  -1]
], dtype=float)

# =====================================================
# DISPLAY DATA
# =====================================================
st.header("📥 Given Correspondences")

c1, c2 = st.columns(2)

with c1:
    st.subheader("3D Points P (homogeneous)")
    st.write(P)

with c2:
    st.subheader("2D Image Points p'")
    st.write(p_img)

# =====================================================
# BUILD DLT SYSTEM
# =====================================================
st.divider()
st.header("🧮 Build Linear System (DLT)")

def build_dlt_matrix(P, p):
    """
    Builds matrix A for Ap = 0
    """
    A = []

    for i in range(len(P)):
        X, Y, Z, W = P[i]
        x, y = p[i]

        row1 = [ X, Y, Z, W, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x*W ]
        row2 = [ 0, 0, 0, 0, X, Y, Z, W, -y*X, -y*Y, -y*Z, -y*W ]

        A.append(row1)
        A.append(row2)

    return np.array(A, dtype=float)

A = build_dlt_matrix(P, p_img)

st.subheader("DLT Matrix A")
st.write(A)

# =====================================================
# SOLVE USING SVD
# =====================================================
st.divider()
st.header("📐 Solve for Projection Matrix C'")

U, S, Vt = np.linalg.svd(A)
c = Vt[-1]                 # smallest singular value
C_prime = c.reshape(3, 4)

# Normalize (scale ambiguity)
C_prime = C_prime / C_prime[-1, -1]

st.subheader("Estimated Projection Matrix C'")
st.write(C_prime)

# =====================================================
# VERIFICATION
# =====================================================
st.divider()
st.header("✅ Reprojection Verification")

def project_points(P, C):
    """
    Project homogeneous 3D points using camera matrix C.
    """
    P_proj = (C @ P.T).T
    P_proj = P_proj[:, :2] / P_proj[:, [2]]
    return P_proj

p_est = project_points(P, C_prime)

st.subheader("Predicted Image Points")
st.write(p_est)

st.subheader("Measured Image Points")
st.write(p_img)

errors = np.linalg.norm(p_est - p_img, axis=1)
rmse = np.sqrt(np.mean(errors**2))

st.write("Reprojection errors per point:")
st.write(errors)

st.success("RMSE = " + str(round(rmse, 6)))

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Visualization")

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(p_img[:,0], p_img[:,1], s=120, label="Observed p'")
ax.scatter(p_est[:,0], p_est[:,1], s=120, marker="x", label="Projected p̂'")

for i in range(len(p_img)):
    ax.plot(
        [p_img[i,0], p_est[i,0]],
        [p_img[i,1], p_est[i,1]],
        "--",
        alpha=0.5
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Camera Calibration Reprojection")
ax.grid(True)
ax.legend()
ax.set_aspect("equal", adjustable="box")

st.pyplot(fig)

# =====================================================
# EXPLANATION
# =====================================================
st.divider()
st.header("🧠 Explanation")

st.markdown("""
### Direct Linear Transform (DLT)

Each correspondence provides two linear equations:

x (c3ᵀ X) = c1ᵀ X  
y (c3ᵀ X) = c2ᵀ X  

Stacking all correspondences yields:

A c = 0

The solution is the right singular vector corresponding to the smallest singular value of A.

Because the projection matrix is defined up to scale, we normalize it.

---

### Why SVD?

- Robust to noise
- Finds least-squares solution
- Standard camera calibration technique

---

### Validation

We project all 3D points using C' and compare them to measured image points.
Small RMSE indicates correct calibration.
""")

# =====================================================
# OPTIONAL INTERACTIVE TEST
# =====================================================
st.divider()
st.header("🎛 Test a New 3D Point")

tx = st.slider("X", -2.0, 2.0, 0.5, 0.1)
ty = st.slider("Y", -2.0, 2.0, 0.5, 0.1)
tz = st.slider("Z",  1.0, 5.0, 3.0, 0.1)

P_test = np.array([[tx, ty, tz, 1.0]])
p_test = project_points(P_test, C_prime)

st.write("3D point:", P_test)
st.write("Projected image point:", p_test)

st.caption("🚀 Camera calibration using Direct Linear Transform (DLT).")
