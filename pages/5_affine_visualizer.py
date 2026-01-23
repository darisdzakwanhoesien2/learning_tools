import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.io import save_run, load_runs, clear_runs

st.set_page_config(layout="wide")
st.title("📐 Affine Least Squares Visualizer")

# =====================================================
# Utilities
# =====================================================

def format_dot_product(row, vec):
    """
    Create step-by-step dot product string:
    (a·b) + (c·d) + ...
    """
    terms = []
    for a, b in zip(row, vec.flatten()):
        terms.append(f"({a:.0f}·{b:.0f})")
    expr = " + ".join(terms)
    value = np.dot(row, vec.flatten())
    return expr, value


# =====================================================
# Load History
# =====================================================

runs = load_runs()

selected_run = None
if runs:
    st.sidebar.subheader("📜 Load Previous Run")

    labels_runs = [
        f"{i+1} — {r['timestamp']}"
        for i, r in enumerate(runs)
    ]

    idx = st.sidebar.selectbox(
        "Select run",
        options=list(range(len(runs))),
        format_func=lambda i: labels_runs[i]
    )

    selected_run = runs[idx]

# =====================================================
# Defaults
# =====================================================

labels = ["A", "B", "C", "D"]

default_src = np.array([
    [-1,  1],
    [ 1,  1],
    [ 1, -1],
    [-1, -1]
], dtype=float)

default_dst = np.array([
    [ 1, 2],
    [ 3, 2],
    [-1, 0],
    [-3, 0]
], dtype=float)

# Override defaults if loading a run
if selected_run:
    default_src = np.array(selected_run["source"])
    default_dst = np.array(selected_run["target"])

# =====================================================
# Sidebar Inputs
# =====================================================

st.sidebar.header("📌 Edit Points")

def edit_points(name, pts):
    out = []
    st.sidebar.subheader(name)
    for i,p in enumerate(pts):
        x = st.sidebar.number_input(
            f"{name} {labels[i]} x",
            value=float(p[0]),
            key=f"{name}{i}x"
        )
        y = st.sidebar.number_input(
            f"{name} {labels[i]} y",
            value=float(p[1]),
            key=f"{name}{i}y"
        )
        out.append([x,y])
    return np.array(out)

src = edit_points("Source", default_src)
dst = edit_points("Target", default_dst)

# =====================================================
# Point Table
# =====================================================

st.subheader("📌 Point Correspondences")

table = pd.DataFrame({
    "Point": labels,
    "Source (x, y)": [tuple(p) for p in src],
    "Target (x', y')": [tuple(p) for p in dst],
    "Mapping": [
        f"{tuple(src[i])} → {tuple(dst[i])}"
        for i in range(len(src))
    ]
})

st.dataframe(table, use_container_width=True)

# =====================================================
# Linear System
# =====================================================

N = src.shape[0]
M = np.hstack([src, np.ones((N,1))])
bx = dst[:,0].reshape(-1,1)
by = dst[:,1].reshape(-1,1)

st.subheader("🧮 Linear System")

c1, c2, c3 = st.columns(3)
c1.dataframe(pd.DataFrame(M, columns=["x","y","1"]))
c2.dataframe(pd.DataFrame(bx, columns=["bₓ"]))
c3.dataframe(pd.DataFrame(by, columns=["bᵧ"]))

# =====================================================
# Least Squares
# =====================================================

MT = M.T
MTM = MT @ M
MTM_inv = np.linalg.inv(MTM)
MTbx = MT @ bx
MTby = MT @ by

ax = MTM_inv @ MTbx
ay = MTM_inv @ MTby

st.subheader("🔢 Least Squares Matrices")

c1, c2 = st.columns(2)
c1.markdown("### Mᵀ M")
c1.dataframe(pd.DataFrame(MTM))

c2.markdown("### (Mᵀ M)⁻¹")
c2.dataframe(pd.DataFrame(MTM_inv))

# =====================================================
# Step-by-step MTb Explanation
# =====================================================

st.subheader("✏️ Step-by-Step Computation of Mᵀ b")

st.markdown("### 🔹 For bₓ")

for i in range(MT.shape[0]):
    expr, value = format_dot_product(MT[i], bx)
    st.latex(rf"M^T b_x [{i}] = {expr} = {value:.0f}")

st.markdown("### 🔹 For bᵧ")

for i in range(MT.shape[0]):
    expr, value = format_dot_product(MT[i], by)
    st.latex(rf"M^T b_y [{i}] = {expr} = {value:.0f}")

# =====================================================
# Final Affine Matrix
# =====================================================

H = np.array([
    [ax[0,0], ax[1,0], ax[2,0]],
    [ay[0,0], ay[1,0], ay[2,0]],
    [0,0,1]
])

st.subheader("✅ Final Affine Matrix")

st.latex(rf"""
H_{{affine}} =
\begin{{bmatrix}}
{H[0,0]:.2f} & {H[0,1]:.2f} & {H[0,2]:.2f} \\
{H[1,0]:.2f} & {H[1,1]:.2f} & {H[1,2]:.2f} \\
0 & 0 & 1
\end{{bmatrix}}
""")

st.dataframe(pd.DataFrame(H))

# =====================================================
# Visualization
# =====================================================

def apply_transform(points, H):
    pts_h = np.hstack([points, np.ones((points.shape[0],1))])
    out = (H @ pts_h.T).T
    return out[:,:2]

pred = apply_transform(src, H)

st.subheader("📊 Geometric Visualization")

fig, axp = plt.subplots(figsize=(6,6))
axp.scatter(src[:,0], src[:,1], label="Source")
axp.scatter(dst[:,0], dst[:,1], label="Target")
axp.scatter(pred[:,0], pred[:,1], marker="x", label="Prediction")
axp.legend()
axp.set_aspect("equal")
st.pyplot(fig)

# =====================================================
# Storage Controls
# =====================================================

st.subheader("💾 Storage")

payload = {
    "source": src.tolist(),
    "target": dst.tolist(),
    "H_affine": H.tolist(),
    "ax": ax.flatten().tolist(),
    "ay": ay.flatten().tolist()
}

c1, c2, c3 = st.columns(3)

if c1.button("💾 Save Run"):
    save_run(payload)
    st.success("Run saved successfully!")

if c2.button("🧹 Clear History"):
    clear_runs()
    st.warning("History cleared.")

# =====================================================
# History Viewer
# =====================================================

st.subheader("📜 Run History")

runs = load_runs()

if runs:
    hist_df = pd.DataFrame([
        {
            "timestamp": r["timestamp"],
            "H_affine": r["H_affine"]
        }
        for r in runs
    ])
    st.dataframe(hist_df, use_container_width=True)
else:
    st.info("No saved runs yet.")


# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from core.io import save_run, load_runs, clear_runs

# st.set_page_config(layout="wide")
# st.title("📐 Affine Least Squares Visualizer")

# # =====================================================
# # Defaults
# # =====================================================

# labels = ["A", "B", "C", "D"]

# default_src = np.array([
#     [-1,  1],
#     [ 1,  1],
#     [ 1, -1],
#     [-1, -1]
# ], dtype=float)

# default_dst = np.array([
#     [ 1, 2],
#     [ 3, 2],
#     [-1, 0],
#     [-3, 0]
# ], dtype=float)

# # =====================================================
# # Sidebar Inputs
# # =====================================================

# st.sidebar.header("📌 Edit Points")

# def edit_points(name, pts):
#     out = []
#     st.sidebar.subheader(name)
#     for i,p in enumerate(pts):
#         x = st.sidebar.number_input(
#             f"{name} {labels[i]} x",
#             value=float(p[0]),
#             key=f"{name}{i}x"
#         )
#         y = st.sidebar.number_input(
#             f"{name} {labels[i]} y",
#             value=float(p[1]),
#             key=f"{name}{i}y"
#         )
#         out.append([x,y])
#     return np.array(out)

# src = edit_points("Source", default_src)
# dst = edit_points("Target", default_dst)

# # =====================================================
# # Tables
# # =====================================================

# st.subheader("📌 Point Correspondences")

# table = pd.DataFrame({
#     "Point": labels,
#     "Source (x, y)": [tuple(p) for p in src],
#     "Target (x', y')": [tuple(p) for p in dst],
# })

# st.dataframe(table, use_container_width=True)

# # =====================================================
# # Build Linear System
# # =====================================================

# N = src.shape[0]
# M = np.hstack([src, np.ones((N,1))])
# bx = dst[:,0].reshape(-1,1)
# by = dst[:,1].reshape(-1,1)

# st.subheader("🧮 Linear System")

# c1, c2, c3 = st.columns(3)
# c1.dataframe(pd.DataFrame(M, columns=["x","y","1"]))
# c2.dataframe(pd.DataFrame(bx, columns=["x'"]))
# c3.dataframe(pd.DataFrame(by, columns=["y'"]))

# # =====================================================
# # Least Squares
# # =====================================================

# MT = M.T
# MTM = MT @ M
# MTM_inv = np.linalg.inv(MTM)
# MTbx = MT @ bx
# MTby = MT @ by
# ax = MTM_inv @ MTbx
# ay = MTM_inv @ MTby

# st.subheader("🔢 Least Squares Matrices")

# c1, c2 = st.columns(2)
# c1.dataframe(pd.DataFrame(MTM))
# c2.dataframe(pd.DataFrame(MTM_inv))

# # =====================================================
# # Final Matrix
# # =====================================================

# H = np.array([
#     [ax[0,0], ax[1,0], ax[2,0]],
#     [ay[0,0], ay[1,0], ay[2,0]],
#     [0,0,1]
# ])

# st.subheader("✅ H_affine")

# st.latex(rf"""
# H =
# \begin{{bmatrix}}
# {H[0,0]:.2f} & {H[0,1]:.2f} & {H[0,2]:.2f} \\
# {H[1,0]:.2f} & {H[1,1]:.2f} & {H[1,2]:.2f} \\
# 0 & 0 & 1
# \end{{bmatrix}}
# """)

# # =====================================================
# # Visualization
# # =====================================================

# def apply_transform(points, H):
#     pts_h = np.hstack([points, np.ones((points.shape[0],1))])
#     out = (H @ pts_h.T).T
#     return out[:,:2]

# pred = apply_transform(src, H)

# fig, axp = plt.subplots(figsize=(6,6))
# axp.scatter(src[:,0], src[:,1], label="Source")
# axp.scatter(dst[:,0], dst[:,1], label="Target")
# axp.scatter(pred[:,0], pred[:,1], marker="x", label="Prediction")
# axp.legend()
# axp.set_aspect("equal")
# st.pyplot(fig)

# # =====================================================
# # Persistence Controls
# # =====================================================

# st.subheader("💾 Storage")

# payload = {
#     "source": src.tolist(),
#     "target": dst.tolist(),
#     "H_affine": H.tolist(),
#     "ax": ax.flatten().tolist(),
#     "ay": ay.flatten().tolist()
# }

# c1, c2, c3 = st.columns(3)

# if c1.button("💾 Save Run"):
#     save_run(payload)
#     st.success("Run saved successfully!")

# if c2.button("🧹 Clear History"):
#     clear_runs()
#     st.warning("History cleared.")

# # =====================================================
# # History Viewer
# # =====================================================

# st.subheader("📜 Run History")

# runs = load_runs()

# if runs:
#     df = pd.DataFrame([
#         {
#             "timestamp": r["timestamp"],
#             "H_affine": r["H_affine"]
#         }
#         for r in runs
#     ])
#     st.dataframe(df, use_container_width=True)
# else:
#     st.info("No saved runs yet.")


# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# from core.transforms import (
#     estimate_affine_ls,
#     estimate_homography,
#     apply_transform
# )

# st.set_page_config(layout="wide")
# st.title("📐 Affine Transformation Estimation & Model Comparison")

# st.markdown("""
# This app demonstrates:

# ✅ Least Squares Affine Estimation  
# ✅ Matrix Construction (M, MᵀM, inverse)  
# ✅ Point Mapping Visualization  
# ✅ Affine vs Homography Comparison  
# """)

# # ------------------------------------------------------
# # Input Points
# # ------------------------------------------------------

# st.sidebar.header("📌 Input Point Correspondences")

# default_src = np.array([
#     [-1,  1],
#     [ 1,  1],
#     [ 1, -1],
#     [-1, -1]
# ], dtype=float)

# default_dst = np.array([
#     [ 1, 2],
#     [ 3, 2],
#     [-1, 0],
#     [-3, 0]
# ], dtype=float)

# def edit_points(label, pts):
#     st.sidebar.subheader(label)
#     out = []
#     for i,p in enumerate(pts):
#         x = st.sidebar.number_input(f"{label} P{i+1} x", value=float(p[0]), key=f"{label}{i}x")
#         y = st.sidebar.number_input(f"{label} P{i+1} y", value=float(p[1]), key=f"{label}{i}y")
#         out.append([x,y])
#     return np.array(out)

# src = edit_points("Source", default_src)
# dst = edit_points("Target", default_dst)

# # ------------------------------------------------------
# # Estimation
# # ------------------------------------------------------

# H_affine, dbg = estimate_affine_ls(src, dst)
# H_homo = estimate_homography(src, dst)

# pred_affine = apply_transform(src, H_affine)
# pred_homo   = apply_transform(src, H_homo)

# # ------------------------------------------------------
# # Visualization
# # ------------------------------------------------------

# st.subheader("📊 Point Mapping Visualization")

# fig, ax = plt.subplots(figsize=(7,7))

# ax.scatter(src[:,0], src[:,1], c="blue", label="Source")
# ax.scatter(dst[:,0], dst[:,1], c="green", label="Target")
# ax.scatter(pred_affine[:,0], pred_affine[:,1], 
#            c="red", marker="x", label="Affine Prediction")
# ax.scatter(pred_homo[:,0], pred_homo[:,1], 
#            c="purple", marker="+", label="Homography Prediction")

# for i in range(len(src)):
#     ax.plot([src[i,0], pred_affine[i,0]],
#             [src[i,1], pred_affine[i,1]], 'r--', alpha=0.5)

# ax.axhline(0,color="gray",alpha=0.3)
# ax.axvline(0,color="gray",alpha=0.3)
# ax.set_aspect("equal")
# ax.legend()
# st.pyplot(fig)

# # ------------------------------------------------------
# # Matrix Inspection
# # ------------------------------------------------------

# st.subheader("🧮 Least Squares Matrices")

# c1, c2 = st.columns(2)

# with c1:
#     st.markdown("### Design Matrix M")
#     st.code(dbg["M"])

#     st.markdown("### MᵀM")
#     st.code(dbg["MtM"])

# with c2:
#     st.markdown("### (MᵀM)⁻¹")
#     st.code(dbg["MtM_inv"])

#     st.markdown("### Parameters")
#     st.write("a₁ a₂ a₃ =", dbg["ax"])
#     st.write("a₄ a₅ a₆ =", dbg["ay"])

# st.markdown("### ✅ Affine Matrix")
# st.latex(rf"""
# H =
# \begin{{bmatrix}}
# {dbg["ax"][0]:.2f} & {dbg["ax"][1]:.2f} & {dbg["ax"][2]:.2f} \\
# {dbg["ay"][0]:.2f} & {dbg["ay"][1]:.2f} & {dbg["ay"][2]:.2f} \\
# 0 & 0 & 1
# \end{{bmatrix}}
# """)

# # ------------------------------------------------------
# # Error Metrics
# # ------------------------------------------------------

# affine_err = np.linalg.norm(dst - pred_affine, axis=1).mean()
# homo_err   = np.linalg.norm(dst - pred_homo, axis=1).mean()

# st.subheader("📏 Mean Reprojection Error")

# c1, c2 = st.columns(2)
# c1.metric("Affine Error", f"{affine_err:.4f}")
# c2.metric("Homography Error", f"{homo_err:.4f}")

# # ------------------------------------------------------
# # Explainability Section
# # ------------------------------------------------------

# st.subheader("🧠 Model Comparison")

# st.markdown("""
# ### 🔹 Affine Model
# - 6 Degrees of Freedom
# - Preserves parallel lines
# - Robust for weak perspective
# - Less sensitive to noise

# ### 🔹 Homography Model
# - 8 Degrees of Freedom
# - Models perspective distortion
# - Can overfit with small datasets
# - Requires at least 4 correspondences

# ### ✅ Practical Guidance
# Use **Affine** when:
# - Object is far from camera
# - Scene is approximately planar
# - Perspective distortion is small

# Use **Homography** when:
# - Strong perspective effects exist
# - Plane is tilted significantly
# - Camera is close to object
# """)