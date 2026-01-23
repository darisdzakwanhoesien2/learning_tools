import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Color Histogram Intersection", layout="wide")
st.title("🎨 Color Histograms & Histogram Intersection")

# =====================================================
# GIVEN DATA
# =====================================================

# Model image M (2×3)
MR = np.array([[100,100,100],
               [  0,  0,  0]])
MG = np.array([[  0,  0,  0],
               [100,100,100]])
MB = np.zeros((2,3))

# Image I (4×5)
IR = np.array([
    [0,0,0,0,0],
    [0,150,150,150,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
])

IG = np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,100,100,100,0],
    [0,0,0,0,0]
])

IB = np.array([
    [40,40,40,40,40],
    [30, 0, 0, 0,30],
    [20, 0, 0, 0,20],
    [10,10,10,10,10]
])

# =====================================================
# DISPLAY IMAGES
# =====================================================

st.header("📥 Input Images")

def show_matrix(title, mat):
    st.markdown(f"**{title}**")
    st.dataframe(mat)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model M (2×3)")
    show_matrix("M_R", MR)
    show_matrix("M_G", MG)
    show_matrix("M_B", MB)

with col2:
    st.subheader("Image I (4×5)")
    show_matrix("I_R", IR)
    show_matrix("I_G", IG)
    show_matrix("I_B", IB)

# =====================================================
# STEP 1 — NORMALIZE TO (r,g)
# =====================================================

st.divider()
st.header("🧮 Step 1 — Normalize to (r, g)")

st.latex(r"""
r = \frac{R}{R+G+B}, \quad
g = \frac{G}{R+G+B}
""")

def normalize_rg(R, G, B):
    S = R + G + B
    r = np.zeros_like(R, dtype=float)
    g = np.zeros_like(G, dtype=float)
    mask = S > 0
    r[mask] = R[mask] / S[mask]
    g[mask] = G[mask] / S[mask]
    return r, g

Mr, Mg = normalize_rg(MR, MG, MB)
Ir, Ig = normalize_rg(IR, IG, IB)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model (r,g)")
    st.write("r:")
    st.dataframe(Mr)
    st.write("g:")
    st.dataframe(Mg)

with col2:
    st.subheader("Image (r,g)")
    st.write("r:")
    st.dataframe(Ir)
    st.write("g:")
    st.dataframe(Ig)

# =====================================================
# STEP 2 — HISTOGRAM BINS
# =====================================================

st.divider()
st.header("🧮 Step 2 — Histogram Binning")

st.markdown("""
Bins:

- **B1:** g ≥ 0.5  
- **B2:** r < 0.5 and g < 0.5  
- **B3:** r ≥ 0.5  
""")

def compute_histogram(r, g):
    B1 = np.sum(g >= 0.5)
    B2 = np.sum((r < 0.5) & (g < 0.5))
    B3 = np.sum(r >= 0.5)
    return np.array([B1, B2, B3])

H_M = compute_histogram(Mr, Mg)
H_I = compute_histogram(Ir, Ig)

st.write("Histogram of Model M:")
st.write(H_M)

st.write("Histogram of Image I:")
st.write(H_I)

# =====================================================
# STEP 3 — HISTOGRAM INTERSECTION
# =====================================================

st.divider()
st.header("📏 Step 3 — Histogram Intersection")

intersection = np.minimum(H_M, H_I)
match = intersection.sum() / H_M.sum()

st.write("Intersection:")
st.write(intersection)

st.latex(fr"""
\text{{Match}} =
\frac{{\sum \min(H_M, H_I)}}{{\sum H_M}}
= {match:.3f}
""")

st.success(f"✅ Match value = {match:.3f}")

# =====================================================
# STEP 4 — BACKGROUND MODIFICATION
# =====================================================

st.divider()
st.header("🎛 Step 4 — Background Becomes Green")

IG_green = IG.copy()
IB_green = IB.copy()
IR_green = IR.copy()

# Set boundary pixels to green
mask_bg = (IB > 0)
IG_green[mask_bg] = 100
IR_green[mask_bg] = 0
IB_green[mask_bg] = 0

Ir_g, Ig_g = normalize_rg(IR_green, IG_green, IB_green)
H_I_green = compute_histogram(Ir_g, Ig_g)

intersection_green = np.minimum(H_M, H_I_green)
match_green = intersection_green.sum() / H_M.sum()

st.write("New Histogram of I (Green Background):")
st.write(H_I_green)

st.latex(fr"""
\text{{New Match}} = {match_green:.3f}
""")

# =====================================================
# DISCUSSION
# =====================================================

st.divider()
st.header("🧠 Discussion")

st.markdown("""
### 📈 When does background increase the match?

The match increases when background pixels fall into the same histogram bins as the model colors.

In this problem:
- Model contains strong **red** and **green** bins.
- Turning background green increases occupancy in **B1 (g ≥ 0.5)**.
- This increases histogram overlap.

### ⚠️ Risk:
Histogram-based matching is sensitive to background clutter.
""")

st.caption("🚀 Fully reproducible histogram intersection pipeline.")
