import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Filter Bank Texture Classification", layout="wide")
st.title("🧩 Filter Bank Texture Classification")

# =====================================================
# INPUT PATCHES
# =====================================================

A = np.array([
    [1,3,1,3],
    [2,3,2,4],
    [1,3,1,4],
    [1,2,1,4]
], dtype=float)

B = np.array([
    [3,2,3,3],
    [1,1,1,2],
    [3,4,4,4],
    [1,2,1,1]
], dtype=float)

U = np.array([
    [1,2,1,1],
    [4,3,4,3],
    [2,1,2,1],
    [2,3,4,2]
], dtype=float)

st.header("📥 Image Patches")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Patch A")
    st.dataframe(A)
with col2:
    st.subheader("Patch B")
    st.dataframe(B)
with col3:
    st.subheader("Unknown Patch")
    st.dataframe(U)

# =====================================================
# FILTER MASKS
# =====================================================

M1 = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
], dtype=float)

M2 = np.array([
    [-1,-2,-1],
    [ 0, 0, 0],
    [ 1, 2, 1]
], dtype=float)

M3 = np.array([
    [-1, 2,-1],
    [-2, 4,-2],
    [-1, 2,-1]
], dtype=float)

M4 = np.array([
    [-1,-2,-1],
    [ 2, 4, 2],
    [-1,-2,-1]
], dtype=float)

masks = [M1, M2, M3, M4]

st.header("🧱 Filter Masks")

cols = st.columns(4)
for i, (mask, col) in enumerate(zip(masks, cols), start=1):
    with col:
        st.subheader(f"Mask {i}")
        st.dataframe(mask)

# =====================================================
# CONVOLUTION (VALID)
# =====================================================

def valid_convolution(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = img[i:i+kh, j:j+kw]
            out[i,j] = np.sum(patch * kernel)
    return out

# =====================================================
# FEATURE EXTRACTION
# =====================================================

st.divider()
st.header("🧮 Step 1 — Filter Responses")

def extract_features(img):
    responses = []
    features = []
    for mask in masks:
        r = valid_convolution(img, mask)
        responses.append(r)
        features.append(np.sum(np.abs(r)))
    return responses, np.array(features)

resp_A, feat_A = extract_features(A)
resp_B, feat_B = extract_features(B)
resp_U, feat_U = extract_features(U)

# =====================================================
# DISPLAY FILTER RESPONSES
# =====================================================

def show_responses(name, responses):
    st.subheader(name)
    fig, axes = plt.subplots(1,4, figsize=(12,3))
    for i, ax in enumerate(axes):
        im = ax.imshow(responses[i], cmap="viridis")
        ax.set_title(f"Mask {i+1}")
        ax.axis("off")
        plt.colorbar(im, ax=ax)
    st.pyplot(fig)

show_responses("Patch A Responses", resp_A)
show_responses("Patch B Responses", resp_B)
show_responses("Unknown Patch Responses", resp_U)

# =====================================================
# FEATURE VECTORS
# =====================================================

st.divider()
st.header("📊 Step 2 — Feature Vectors")

st.write("Feature vector A:", feat_A)
st.write("Feature vector B:", feat_B)
st.write("Feature vector Unknown:", feat_U)

st.latex(r"""
f_i = \sum_{x,y} |R_i(x,y)|
""")

# =====================================================
# CLASSIFICATION
# =====================================================

st.divider()
st.header("📏 Step 3 — Classification")

def euclidean(a, b):
    return np.linalg.norm(a - b)

dA = euclidean(feat_U, feat_A)
dB = euclidean(feat_U, feat_B)

st.latex(fr"""
d(U,A) = {dA:.3f}, \quad
d(U,B) = {dB:.3f}
""")

if dA < dB:
    st.success("✅ Unknown patch is closer to **Patch A**")
else:
    st.success("✅ Unknown patch is closer to **Patch B**")

# =====================================================
# EXPLANATION
# =====================================================

st.divider()
st.header("🧠 Explanation")

st.markdown("""
### Why this works:

- Filters detect edges, textures, orientations.
- Absolute sum captures texture energy.
- Feature vectors summarize texture structure.
- Euclidean distance compares similarity.

This is a simple but powerful texture classifier.
""")

st.caption("🚀 Filter bank texture classification complete.")
