import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Harris Corner Detection",
    layout="wide"
)
st.title("🟦 Harris Corner Detection — Step-by-Step")

# =====================================================
# GIVEN IMAGE PATCH (7×7)
# =====================================================
I = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2]
], dtype=float)

# Marked pixel location (x, y) = (4,2) in image coordinates
# Using (row, col) = (2,4) in array indexing
target = (2, 4)

# =====================================================
# PREWITT FILTERS
# =====================================================
Hx = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

Hy = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=float)

alpha = 0.05
threshold = 600

# =====================================================
# DISPLAY INPUT
# =====================================================
st.header("📥 Given Image Patch I(x, y)")
st.write(I)

st.write("Target pixel (row, col):", target)

st.subheader("Prewitt Masks")
c1, c2 = st.columns(2)
with c1:
    st.write("Hₓ")
    st.write(Hx)
with c2:
    st.write("Hᵧ")
    st.write(Hy)

# =====================================================
# GRADIENT COMPUTATION
# =====================================================
st.divider()
st.header("🧮 Step 1 — Compute Image Gradients")

Ix = np.zeros_like(I)
Iy = np.zeros_like(I)

for r in range(1, I.shape[0]-1):
    for c in range(1, I.shape[1]-1):
        patch = I[r-1:r+2, c-1:c+2]
        Ix[r, c] = np.sum(patch * Hx)
        Iy[r, c] = np.sum(patch * Hy)

st.subheader("Gradient Iₓ")
st.write(Ix)

st.subheader("Gradient Iᵧ")
st.write(Iy)

# =====================================================
# STRUCTURE TENSOR WINDOW (3×3)
# =====================================================
st.divider()
st.header("🧮 Step 2 — Structure Tensor (3×3 Window)")

r, c = target

Ix_w = Ix[r-1:r+2, c-1:c+2]
Iy_w = Iy[r-1:r+2, c-1:c+2]

st.write("Iₓ window:")
st.write(Ix_w)

st.write("Iᵧ window:")
st.write(Iy_w)

# =====================================================
# COMPUTE SECOND MOMENT MATRIX
# =====================================================
st.divider()
st.header("🧮 Step 3 — Second Moment Matrix")

Sxx = np.sum(Ix_w**2)
Syy = np.sum(Iy_w**2)
Sxy = np.sum(Ix_w * Iy_w)

M = np.array([
    [Sxx, Sxy],
    [Sxy, Syy]
])

st.write("Second Moment Matrix M:")
st.write(M)

# =====================================================
# HARRIS RESPONSE
# =====================================================
st.divider()
st.header("🧮 Step 4 — Harris Response")

detM = np.linalg.det(M)
traceM = np.trace(M)

R = detM - alpha * (traceM ** 2)

st.write("det(M):", detM)
st.write("trace(M):", traceM)
st.success("Harris response R = " + str(round(R, 3)))

# =====================================================
# DECISION
# =====================================================
st.divider()
st.header("🎯 Corner Decision")

if R > threshold:
    st.success("✅ R > threshold → CORNER detected")
else:
    st.error("❌ R ≤ threshold → NOT a corner")

st.write("Threshold =", threshold)

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Visualization")

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(I, cmap="gray")
ax.scatter(c, r, color="red", s=120, marker="x", label="Target Pixel")
ax.set_title("Image Patch with Target Pixel")
ax.legend()
ax.set_xticks(range(7))
ax.set_yticks(range(7))
ax.grid(True)

st.pyplot(fig)

# =====================================================
# EXPLANATION
# =====================================================
st.divider()
st.header("🧠 Interpretation")

st.markdown("""
### Harris Corner Logic
- Large **Iₓ²** and **Iᵧ²** → intensity changes in both directions
- Large determinant and trace → strong corner
- Flat regions → small values
- Edges → large in one direction only

### Final Result
The Harris response is compared against the threshold **T = 600**
to decide whether the point is a corner.
""")

st.caption("🚀 Harris corner detection using Prewitt gradients and 3×3 window.")
