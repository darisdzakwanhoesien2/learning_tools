import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

st.set_page_config(page_title="Texture Classification with Roberts Filters", layout="wide")
st.title("🧵 Texture Classification using Roberts Gradient Masks")

# =====================================================
# INPUT PATCHES
# =====================================================

st.header("🖼 Input Image Patches")

class1 = np.array([
    [4,1,4,1],
    [5,1,1,1],
    [3,4,5,3],
    [1,1,2,2]
])

class2 = np.array([
    [1,3,2,1],
    [0,2,2,2],
    [0,5,4,2],
    [4,0,4,3]
])

unknown = np.array([
    [5,2,5,2],
    [2,2,6,2],
    [6,4,4,5],
    [3,3,2,2]
])

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Class 1")
    st.dataframe(class1)
with col2:
    st.subheader("Class 2")
    st.dataframe(class2)
with col3:
    st.subheader("Unknown")
    st.dataframe(unknown)

# =====================================================
# ROBERTS FILTERS
# =====================================================

st.header("🧮 Roberts Gradient Masks")

K1 = np.array([[0, 1],
               [-1, 0]])

K2 = np.array([[1, 0],
               [0, -1]])

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Kernel K1**")
    st.dataframe(K1)
with col2:
    st.markdown("**Kernel K2**")
    st.dataframe(K2)

# =====================================================
# VALID CONVOLUTION
# =====================================================

def valid_conv2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            patch = img[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * kernel)
    return out

def visualize(mat, title):
    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="viridis")
    ax.set_title(title)
    plt.colorbar(im)
    st.pyplot(fig)

# =====================================================
# FILTER RESPONSES
# =====================================================

st.header("📐 Step 1 — Filter Responses (Valid Pixels Only)")

responses = {}

for name, img in {
    "Class 1": class1,
    "Class 2": class2,
    "Unknown": unknown
}.items():

    r1 = valid_conv2d(img, K1)
    r2 = valid_conv2d(img, K2)

    responses[name] = (r1, r2)

    st.subheader(name)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Response K1**")
        st.dataframe(r1)
        visualize(r1, f"{name} — K1 Response")

    with col2:
        st.markdown("**Response K2**")
        st.dataframe(r2)
        visualize(r2, f"{name} — K2 Response")

# =====================================================
# FEATURE VECTORS
# =====================================================

st.header("📊 Step 2 — Feature Vectors (Mean Responses)")

features = {}

for name, (r1, r2) in responses.items():
    fvec = np.array([np.mean(r1), np.mean(r2)])
    features[name] = fvec

    st.latex(fr"{name}: \quad \mu = [{fvec[0]:.3f}, {fvec[1]:.3f}]")

# =====================================================
# DISTANCE + CLASSIFICATION
# =====================================================

st.header("📏 Step 3 — Euclidean Distance Classification")

u = features["Unknown"]
d1 = norm(u - features["Class 1"])
d2 = norm(u - features["Class 2"])

st.latex(fr"d(Unknown, Class1) = {d1:.4f}")
st.latex(fr"d(Unknown, Class2) = {d2:.4f}")

if d1 < d2:
    result = "Class 1"
else:
    result = "Class 2"

st.success(f"✅ Classified Unknown as: **{result}**")

# =====================================================
# LBP THEORY (PART B)
# =====================================================

st.divider()
st.header("📌 (b) LBP-Based Classification — Conceptual Explanation")

st.markdown("""
### 🧠 Procedure using Local Binary Patterns (LBP)

1. For each pixel, compare neighbors with the center pixel.
2. Assign binary values (1 if neighbor ≥ center, else 0).
3. Convert binary pattern into a decimal LBP code.
4. Build a histogram of LBP codes for each patch.
5. Use the histogram as a feature vector.
6. Compare histograms using distance metrics (χ², Euclidean, etc.).
7. Assign the unknown patch to the closest class.

---

### ⚠️ What problem is related to the given image patches?

These patches are extremely small (4×4):

- LBP histograms become unstable and sparse.
- Boundary effects dominate.
- Noise sensitivity increases.
- Texture statistics are unreliable.

This is known as the **small-sample texture problem** or **insufficient spatial support problem**.
""")
