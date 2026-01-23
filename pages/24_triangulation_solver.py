import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Triangulation from Two Views",
    layout="wide"
)
st.title("📐 Triangulation — Recover 3D Points from Two Views")

# =====================================================
# CAMERA MATRICES
# =====================================================
C = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
], dtype=float)

C2 = np.array([
    [1, 0, 0, 1],
    [0, 0, -1, 1],
    [0, 1, 0, 1]
], dtype=float)

# =====================================================
# IMAGE POINTS (HOMOGENEOUS)
# =====================================================
# p_i = (x, y, 1)
p1 = np.array([
    [ 0.0,  0.0, 1.0],
    [ 1.0,  0.0, 1.0],
    [-1/2,  1/2, 1.0],
    [-1/2,  0.0, 1.0],
    [ 1/3,  1/3, 1.0],
    [ 0.0,  1/3, 1.0]
])

p2 = np.array([
    [ 1.0, -2.0, 1.0],
    [ 2.0, -2.0, 1.0],
    [ 0.0, -1/2, 1.0],
    [ 0.0, -1.0, 1.0],
    [ 1.0, -1.0, 1.0],
    [ 1/2, -1.0, 1.0]
])

# =====================================================
# DISPLAY INPUT
# =====================================================
st.header("📥 Camera Matrices")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Camera C")
    st.write(C)

with c2:
    st.subheader("Camera C'")
    st.write(C2)

st.divider()
st.header("📥 Image Correspondences")

st.write("Points in image 1:")
st.write(p1)

st.write("Points in image 2:")
st.write(p2)

# =====================================================
# TRIANGULATION FUNCTION
# =====================================================
def triangulate_point(p, p2, C, C2):
    """
    Linear triangulation using DLT.
    """
    x, y, _  = p
    x2, y2, _ = p2

    A = np.vstack([
        x * C[2]  - C[0],
        y * C[2]  - C[1],
        x2 * C2[2] - C2[0],
        y2 * C2[2] - C2[1]
    ])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]

# =====================================================
# TRIANGULATE ALL POINTS
# =====================================================
st.divider()
st.header("🧮 Triangulated 3D Points")

P_est = []

for i in range(len(p1)):
    X = triangulate_point(p1[i], p2[i], C, C2)
    P_est.append(X)

P_est = np.array(P_est)

st.write(P_est)

# =====================================================
# REPROJECTION CHECK
# =====================================================
st.divider()
st.header("✅ Reprojection Verification")

def project(P, C):
    P_h = np.hstack([P, np.ones((len(P),1))])
    proj = (C @ P_h.T).T
    return proj[:, :2] / proj[:, [2]]

proj1 = project(P_est, C)
proj2 = project(P_est, C2)

st.subheader("Reprojection in Camera 1")
st.write(proj1)

st.subheader("Reprojection in Camera 2")
st.write(proj2)

err1 = np.linalg.norm(proj1 - p1[:, :2], axis=1)
err2 = np.linalg.norm(proj2 - p2[:, :2], axis=1)

st.write("Errors camera 1:")
st.write(err1)

st.write("Errors camera 2:")
st.write(err2)

st.success("Mean reprojection error = " + str(round((err1.mean()+err2.mean())/2, 6)))

# =====================================================
# 3D VISUALIZATION
# =====================================================
st.divider()
st.header("📊 3D Visualization")

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(projection="3d")

ax.scatter(
    P_est[:,0],
    P_est[:,1],
    P_est[:,2],
    s=80,
    marker="o",
    label="Triangulated Points"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Recovered 3D Points")
ax.legend()

st.pyplot(fig)

# =====================================================
# INTERACTIVE TEST
# =====================================================
st.divider()
st.header("🎛 Inspect a Single Point")

idx = st.slider("Point index", 0, len(P_est)-1, 0)

st.write("3D point:")
st.write(P_est[idx])

st.write("Projection in camera 1:")
st.write(proj1[idx])

st.write("Projection in camera 2:")
st.write(proj2[idx])

st.caption("🚀 Linear triangulation from two calibrated cameras.")
