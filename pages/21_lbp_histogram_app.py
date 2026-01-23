import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="LBP Histogram", layout="wide")
st.title("🧩 Local Binary Patterns (LBP) — 8 Neighborhood")

# =====================================================
# INPUT IMAGE
# =====================================================

I = np.array([
    [1,1,1,0,3,3,3,4],
    [1,0,1,0,3,3,3,4],
    [1,0,1,0,3,3,3,4],
    [1,0,1,2,2,3,0,4],
    [1,1,1,2,2,3,0,4],
    [0,0,0,2,2,3,0,4],
    [0,0,0,2,2,0,0,4],
    [0,0,0,4,4,4,4,4],
], dtype=np.uint8)

H, W = I.shape

st.header("📥 Input Image")
st.dataframe(I)

# =====================================================
# LBP PARAMETERS
# =====================================================

st.header("📐 Neighborhood Definition")

st.markdown("""
We use the clockwise neighborhood order starting from top-left:

\[
p = [(-1,-1), (-1,0), (-1,1), (0,1),
     (1,1), (1,0), (1,-1), (0,-1)]
\]
""")

show_offsets = [(-1,-1), (-1,0), (-1,1), (0,1),
                (1,1), (1,0), (1,-1), (0,-1)]

# =====================================================
# LBP COMPUTATION
# =====================================================

def compute_lbp(image):
    H, W = image.shape
    lbp = np.zeros((H-2, W-2), dtype=np.uint8)

    for i in range(1, H-1):
        for j in range(1, W-1):
            center = image[i,j]
            bits = []
            for dy, dx in show_offsets:
                neighbor = image[i+dy, j+dx]
                bits.append(1 if neighbor >= center else 0)

            code = sum(bit << idx for idx, bit in enumerate(bits))
            lbp[i-1, j-1] = code

    return lbp

LBP = compute_lbp(I)

st.divider()
st.header("🧮 Step 1 — LBP Image")

st.dataframe(LBP)

# =====================================================
# HISTOGRAM
# =====================================================

st.divider()
st.header("📊 Step 2 — 256-bin Histogram")

hist = np.zeros(256, dtype=int)

for val in LBP.flatten():
    hist[val] += 1

st.write("Histogram (non-zero bins):")
nonzero_bins = {i:int(v) for i,v in enumerate(hist) if v > 0}
st.json(nonzero_bins)

# =====================================================
# HISTOGRAM VISUALIZATION
# =====================================================

st.divider()
st.header("📈 Histogram Visualization")

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(range(256), hist)
ax.set_xlabel("LBP Code")
ax.set_ylabel("Frequency")
ax.set_title("LBP Histogram (256 bins)")
ax.grid(True)

st.pyplot(fig)

# =====================================================
# PIXEL INSPECTOR
# =====================================================

st.divider()
st.header("🔍 Pixel Inspector")

row = st.slider("Row (1–6)", 1, H-2, 2)
col = st.slider("Column (1–6)", 1, W-2, 2)

center = I[row, col]
bits = []

for dy, dx in show_offsets:
    neighbor = I[row+dy, col+dx]
    bits.append(1 if neighbor >= center else 0)

binary_string = "".join(str(b) for b in bits[::-1])
code = sum(bit << idx for idx, bit in enumerate(bits))

st.write(f"Center pixel value = {center}")
st.write("Neighbor bits (clockwise):", bits)
st.write("Binary code:", binary_string)
st.success(f"LBP code = {code}")

# =====================================================
# THEORY
# =====================================================

st.divider()
st.header("📘 Theory")

st.latex(r"""
\text{LBP}(x_c, y_c) =
\sum_{p=0}^{7} s(I_p - I_c) 2^p
""")

st.latex(r"""
s(x) =
\begin{cases}
1, & x \ge 0 \\
0, & x < 0
\end{cases}
""")

st.markdown("""
### Interpretation

- Encodes local texture patterns.
- Rotation-sensitive.
- Histogram summarizes texture globally.
- Common in face recognition and texture analysis.
""")

st.caption("🚀 LBP histogram computation complete.")
