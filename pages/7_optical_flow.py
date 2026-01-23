import streamlit as st
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optical Flow Visual Explorer", layout="wide")
st.title("🎥 Optical Flow – Visual & Step-by-Step Explorer")

# =====================================================
# STORAGE
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "optical_flow_results.json"

if not DB_PATH.exists():
    DB_PATH.write_text("[]")

def load_db():
    return json.loads(DB_PATH.read_text())

def save_db(records):
    DB_PATH.write_text(json.dumps(records, indent=2))

# =====================================================
# PREWITT FILTERS
# =====================================================

Kx = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

Ky = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# =====================================================
# DEFAULT DATA
# =====================================================

default_t1 = np.array([
    [3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3],
    [3,3,7,3,3,3,3,3],
    [3,7,7,3,3,3,3,3],
    [3,9,9,7,5,3,3,3],
    [3,3,9,9,7,5,3,3],
    [3,3,3,9,9,7,5,3],
    [3,3,3,3,3,3,3,3],
])

default_t2 = np.array([
    [3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3],
    [3,3,7,7,3,3,3,3],
    [3,3,9,7,5,3,3,3],
    [3,3,9,9,7,5,3,3],
    [3,3,3,9,9,7,5,3],
    [3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3],
])

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🎯 Pixel Selection")

p1 = st.sidebar.text_input("Pixel 1 (row,col)", "3,4")
p2 = st.sidebar.text_input("Pixel 2 (row,col)", "4,4")

r1, c1 = [int(v)-1 for v in p1.split(",")]
r2, c2 = [int(v)-1 for v in p2.split(",")]

# =====================================================
# UTILITIES
# =====================================================

def extract_patch(img, r, c):
    return img[r-1:r+2, c-1:c+2]

def prewitt_fx_fy(patch):
    fx_mat = Kx * patch
    fy_mat = Ky * patch
    return fx_mat, fy_mat, fx_mat.sum(), fy_mat.sum()

def draw_heatmap(mat, title, highlight=None):
    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="viridis")
    ax.set_title(title)
    plt.colorbar(im)

    if highlight is not None:
        r, c = highlight
        ax.scatter(c, r, s=200, facecolors='none', edgecolors='red', linewidths=2)

    st.pyplot(fig)

# =====================================================
# VISUALIZATION
# =====================================================

st.header("🧱 Step 1 — Kernels")

col1, col2 = st.columns(2)
with col1:
    st.write("Kx")
    st.dataframe(Kx)
    draw_heatmap(Kx, "Kx Kernel")

with col2:
    st.write("Ky")
    st.dataframe(Ky)
    draw_heatmap(Ky, "Ky Kernel")

st.header("🖼 Step 2 — Input Frames")

col1, col2 = st.columns(2)
with col1:
    st.write("Frame t1")
    st.dataframe(default_t1)
    draw_heatmap(default_t1, "Frame t1", (r1, c1))

with col2:
    st.write("Frame t2")
    st.dataframe(default_t2)
    draw_heatmap(default_t2, "Frame t2", (r1, c1))

st.caption("🔴 Red circle indicates selected pixel p1")

# =====================================================
# PATCH EXTRACTION
# =====================================================

patch1 = extract_patch(default_t1, r1, c1)
patch2 = extract_patch(default_t1, r2, c2)

st.header("🔍 Step 3 — 3×3 Neighborhoods")

col1, col2 = st.columns(2)
with col1:
    st.write(f"Patch around p1 = ({p1})")
    st.dataframe(patch1)
    draw_heatmap(patch1, "Patch p1")

with col2:
    st.write(f"Patch around p2 = ({p2})")
    st.dataframe(patch2)
    draw_heatmap(patch2, "Patch p2")

# =====================================================
# CONVOLUTION
# =====================================================

fx1_mat, fy1_mat, fx1, fy1 = prewitt_fx_fy(patch1)
fx2_mat, fy2_mat, fx2, fy2 = prewitt_fx_fy(patch2)

st.header("🧮 Step 4 — Prewitt Convolution")

col1, col2 = st.columns(2)
with col1:
    st.write("p1 → Kx ⊙ patch")
    st.dataframe(fx1_mat)
    draw_heatmap(fx1_mat, "Elementwise Kx * patch1")
    st.latex(fr"f_x^{(1)} = {fx1}")

    st.write("p1 → Ky ⊙ patch")
    st.dataframe(fy1_mat)
    draw_heatmap(fy1_mat, "Elementwise Ky * patch1")
    st.latex(fr"f_y^{(1)} = {fy1}")

with col2:
    st.write("p2 → Kx ⊙ patch")
    st.dataframe(fx2_mat)
    draw_heatmap(fx2_mat, "Elementwise Kx * patch2")
    st.latex(fr"f_x^{(2)} = {fx2}")

    st.write("p2 → Ky ⊙ patch")
    st.dataframe(fy2_mat)
    draw_heatmap(fy2_mat, "Elementwise Ky * patch2")
    st.latex(fr"f_y^{(2)} = {fy2}")

# =====================================================
# TEMPORAL DERIVATIVE
# =====================================================

ft1 = float(default_t2[r1, c1] - default_t1[r1, c1])
ft2 = float(default_t2[r2, c2] - default_t1[r2, c2])

# =====================================================
# MATRIX TABLE WITH KERNELS AS COLUMNS AND PATCHES AS ROWS
# =====================================================

st.header("📊 Step 4C — Kernel × Patch Result Table")

st.markdown("""
**Columns = Kernels (Kx, Ky)**  
**Rows = Patches (p1, p2)**  
Each cell shows the element-wise multiplication result and its sum.
""")

# ---------- HEADER ROW ----------
header = st.columns([2, 4, 4])

with header[0]:
    st.markdown("### Patch ↓  /  Kernel →")

with header[1]:
    st.markdown("### Kx")
    st.dataframe(Kx, use_container_width=True)

with header[2]:
    st.markdown("### Ky")
    st.dataframe(Ky, use_container_width=True)


# ---------- ROW: p1 ----------
row_p1 = st.columns([2, 4, 4])

with row_p1[0]:
    st.markdown("### p1")
    st.dataframe(patch1, use_container_width=True)

with row_p1[1]:
    st.markdown("**Kx ⊙ p1**")
    st.dataframe(fx1_mat, use_container_width=True)
    draw_heatmap(fx1_mat, "p1 × Kx")
    st.latex(fr"\sum = {fx1}")

with row_p1[2]:
    st.markdown("**Ky ⊙ p1**")
    st.dataframe(fy1_mat, use_container_width=True)
    draw_heatmap(fy1_mat, "p1 × Ky")
    st.latex(fr"\sum = {fy1}")


# ---------- ROW: p2 ----------
row_p2 = st.columns([2, 4, 4])

with row_p2[0]:
    st.markdown("### p2")
    st.dataframe(patch2, use_container_width=True)

with row_p2[1]:
    st.markdown("**Kx ⊙ p2**")
    st.dataframe(fx2_mat, use_container_width=True)
    draw_heatmap(fx2_mat, "p2 × Kx")
    st.latex(fr"\sum = {fx2}")

with row_p2[2]:
    st.markdown("**Ky ⊙ p2**")
    st.dataframe(fy2_mat, use_container_width=True)
    draw_heatmap(fy2_mat, "p2 × Ky")
    st.latex(fr"\sum = {fy2}")


# # =====================================================
# # MATRIX TABLE VISUALIZATION
# # =====================================================

# st.header("📊 Step 4B — Kernel × Patch Matrix Table")

# st.markdown("""
# This table shows how each kernel (**Kx, Ky**) interacts with each pixel neighborhood (**p1, p2**).
# Each cell represents the element-wise multiplication result before summation.
# """)

# col_headers = st.columns([1, 3, 3])
# col_headers[0].markdown("**Patch ↓ / Kernel →**")
# col_headers[1].markdown("**Kx**")
# col_headers[2].markdown("**Ky**")

# # ---------- Row: p1 ----------
# row_p1 = st.columns([1, 3, 3])
# row_p1[0].markdown("### p1")

# with row_p1[1]:
#     st.markdown("**Kx ⊙ patch1**")
#     st.dataframe(fx1_mat)
#     draw_heatmap(fx1_mat, "p1 × Kx")
#     st.latex(fr"\sum = {fx1}")

# with row_p1[2]:
#     st.markdown("**Ky ⊙ patch1**")
#     st.dataframe(fy1_mat)
#     draw_heatmap(fy1_mat, "p1 × Ky")
#     st.latex(fr"\sum = {fy1}")

# # ---------- Row: p2 ----------
# row_p2 = st.columns([1, 3, 3])
# row_p2[0].markdown("### p2")

# with row_p2[1]:
#     st.markdown("**Kx ⊙ patch2**")
#     st.dataframe(fx2_mat)
#     draw_heatmap(fx2_mat, "p2 × Kx")
#     st.latex(fr"\sum = {fx2}")

# with row_p2[2]:
#     st.markdown("**Ky ⊙ patch2**")
#     st.dataframe(fy2_mat)
#     draw_heatmap(fy2_mat, "p2 × Ky")
#     st.latex(fr"\sum = {fy2}")


st.header("⏱ Step 5 — Temporal Derivatives")

st.latex(fr"f_t^{(1)} = I_2({p1}) - I_1({p1}) = {ft1}")
st.latex(fr"f_t^{(2)} = I_2({p2}) - I_1({p2}) = {ft2}")

# =====================================================
# LINEAR SYSTEM
# =====================================================

A = np.array([[fx1, fy1], [fx2, fy2]])
b = np.array([-ft1, -ft2])
v = np.linalg.solve(A, b)


st.header("📐 Step 6 — Linear System")

st.latex(r"""
\begin{bmatrix}
f_x^{(1)} & f_y^{(1)}\\
f_x^{(2)} & f_y^{(2)}
\end{bmatrix}
\begin{bmatrix}
v_x\\v_y
\end{bmatrix}
=
\begin{bmatrix}
-f_t^{(1)}\\
-f_t^{(2)}
\end{bmatrix}
""")

st.write("A =")
st.write(A)

st.write("b =")
st.write(b)

st.success(f"Flow vector → vx = {v[0]:.4f},  vy = {v[1]:.4f}")




# =====================================================
# SAVE RESULT
# =====================================================

if st.button("💾 Save Result"):
    records = load_db()

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "p1": [r1+1, c1+1],
        "p2": [r2+1, c2+1],
        "fx1": fx1, "fy1": fy1, "ft1": ft1,
        "fx2": fx2, "fy2": fy2, "ft2": ft2,
        "A": A.tolist(),
        "b": b.tolist(),
        "vx": float(v[0]),
        "vy": float(v[1]),
    }

    records.append(record)
    save_db(records)

    st.success("✅ Result saved!")

# =====================================================
# HISTORY
# =====================================================

st.divider()
st.header("📂 Stored Experiments")

records = load_db()

if records:
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)

    idx = st.selectbox("Select record", range(len(records)))
    st.json(records[idx])
else:
    st.info("No stored experiments yet.")


# import streamlit as st
# import numpy as np
# import json
# from pathlib import Path
# from datetime import datetime
# import pandas as pd

# st.set_page_config(page_title="Optical Flow Explorer", layout="wide")
# st.title("🎥 Optical Flow Estimation with Storage")

# # =====================================================
# # STORAGE PATH
# # =====================================================

# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = BASE_DIR / "data"
# DATA_DIR.mkdir(exist_ok=True)

# DB_PATH = DATA_DIR / "optical_flow_results.json"

# if not DB_PATH.exists():
#     DB_PATH.write_text("[]")

# # =====================================================
# # LOAD DATABASE
# # =====================================================

# def load_db():
#     return json.loads(DB_PATH.read_text())

# def save_db(records):
#     DB_PATH.write_text(json.dumps(records, indent=2))

# # =====================================================
# # PREWITT MASKS
# # =====================================================

# Kx = np.array([
#     [-1, 0, 1],
#     [-1, 0, 1],
#     [-1, 0, 1]
# ])

# Ky = np.array([
#     [-1, -1, -1],
#     [ 0,  0,  0],
#     [ 1,  1,  1]
# ])

# # =====================================================
# # DEFAULT DATA
# # =====================================================

# default_t1 = np.array([
#     [3,3,3,3,3,3,3,3],
#     [3,3,3,3,3,3,3,3],
#     [3,3,7,3,3,3,3,3],
#     [3,7,7,3,3,3,3,3],
#     [3,9,9,7,5,3,3,3],
#     [3,3,9,9,7,5,3,3],
#     [3,3,3,9,9,7,5,3],
#     [3,3,3,3,3,3,3,3],
# ])

# default_t2 = np.array([
#     [3,3,3,3,3,3,3,3],
#     [3,3,3,3,3,3,3,3],
#     [3,3,7,7,3,3,3,3],
#     [3,3,9,7,5,3,3,3],
#     [3,3,9,9,7,5,3,3],
#     [3,3,3,9,9,7,5,3],
#     [3,3,3,3,3,3,3,3],
#     [3,3,3,3,3,3,3,3],
# ])

# # =====================================================
# # SIDEBAR INPUTS
# # =====================================================

# st.sidebar.header("📥 Pixel Selection")

# p1 = st.sidebar.text_input("Pixel 1 (row,col)", "3,4")
# p2 = st.sidebar.text_input("Pixel 2 (row,col)", "4,4")

# r1, c1 = [int(v)-1 for v in p1.split(",")]
# r2, c2 = [int(v)-1 for v in p2.split(",")]

# # =====================================================
# # UTILS
# # =====================================================

# def extract_patch(img, r, c):
#     return img[r-1:r+2, c-1:c+2]

# def prewitt_derivatives(patch):
#     fx = float(np.sum(Kx * patch))
#     fy = float(np.sum(Ky * patch))
#     return fx, fy

# # =====================================================
# # COMPUTATION
# # =====================================================

# try:
#     patch1 = extract_patch(default_t1, r1, c1)
#     patch2 = extract_patch(default_t1, r2, c2)

#     fx1, fy1 = prewitt_derivatives(patch1)
#     fx2, fy2 = prewitt_derivatives(patch2)

#     ft1 = float(default_t2[r1, c1] - default_t1[r1, c1])
#     ft2 = float(default_t2[r2, c2] - default_t1[r2, c2])

#     A = np.array([[fx1, fy1], [fx2, fy2]])
#     b = np.array([-ft1, -ft2])

#     v = np.linalg.solve(A, b)

#     success = True
# except Exception as e:
#     success = False
#     st.error(f"❌ Computation error: {e}")

# # =====================================================
# # DISPLAY
# # =====================================================

# st.header("🧩 Computation Result")

# if success:

#     st.write("### Spatial & Temporal Derivatives")

#     df_deriv = pd.DataFrame([
#         {"Pixel": p1, "fx": fx1, "fy": fy1, "ft": ft1},
#         {"Pixel": p2, "fx": fx2, "fy": fy2, "ft": ft2},
#     ])

#     st.dataframe(df_deriv, use_container_width=True)

#     st.write("### Linear System")

#     st.write("A =")
#     st.write(A)

#     st.write("b =")
#     st.write(b)

#     st.success(f"Estimated Flow →  vx = {v[0]:.4f} ,  vy = {v[1]:.4f}")

#     # =====================================================
#     # SAVE BUTTON
#     # =====================================================

#     if st.button("💾 Save Result"):
#         records = load_db()

#         record = {
#             "timestamp": datetime.utcnow().isoformat(),
#             "pixel1": [r1+1, c1+1],
#             "pixel2": [r2+1, c2+1],
#             "fx1": fx1,
#             "fy1": fy1,
#             "ft1": ft1,
#             "fx2": fx2,
#             "fy2": fy2,
#             "ft2": ft2,
#             "A": A.tolist(),
#             "b": b.tolist(),
#             "vx": float(v[0]),
#             "vy": float(v[1]),
#         }

#         records.append(record)
#         save_db(records)

#         st.success("✅ Result saved successfully!")

# # =====================================================
# # RETRIEVE STORED RESULTS
# # =====================================================

# st.divider()
# st.header("📂 Stored Experiments")

# records = load_db()

# if len(records) == 0:
#     st.info("No saved experiments yet.")
# else:
#     df = pd.DataFrame(records)
#     st.dataframe(df, use_container_width=True)

#     selected_idx = st.selectbox(
#         "Select experiment to inspect:",
#         options=list(range(len(records))),
#         format_func=lambda i: f"{i} | {records[i]['timestamp']}"
#     )

#     selected = records[selected_idx]

#     st.subheader("🔍 Selected Record")
#     st.json(selected)
