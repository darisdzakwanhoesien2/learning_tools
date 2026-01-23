import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Color Distance Explorer", layout="wide")
st.title("🎨 Color Distances — XYZ and CIE u′v′")

# =====================================================
# GIVEN COLORS
# =====================================================
C_ref = np.array([10, 170, 75], dtype=float)
C_a   = np.array([40, 130,110], dtype=float)
C_b   = np.array([10, 140, 50], dtype=float)

# RGB → XYZ matrix
M = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505]
])

# =====================================================
# THEORY
# =====================================================
st.header("📘 Problem Formulation")

st.markdown("### Given RGB colors")
st.latex(r"C_{ref} = (10,170,75), \quad C_a = (40,130,110), \quad C_b = (10,140,50)")

st.markdown("### RGB → XYZ mapping")
st.latex(
    r"\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = "
    r"\begin{bmatrix}"
    r"0.4124 & 0.3576 & 0.1805\\"
    r"0.2126 & 0.7152 & 0.0722\\"
    r"0.0193 & 0.1192 & 0.9505"
    r"\end{bmatrix}"
    r"\begin{bmatrix} R \\ G \\ B \end{bmatrix}"
)

st.markdown("### XYZ → CIE $u'v'$ mapping")
st.latex(r"u' = \frac{4X}{X + 15Y + 3Z}, \quad v' = \frac{9Y}{X + 15Y + 3Z}")

# =====================================================
# STEP 1 — RGB → XYZ
# =====================================================
st.divider()
st.header("🧮 Step 1 — Convert RGB → XYZ")

def rgb_to_xyz(rgb):
    return M @ rgb

XYZ_ref = rgb_to_xyz(C_ref)
XYZ_a   = rgb_to_xyz(C_a)
XYZ_b   = rgb_to_xyz(C_b)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("C_ref (XYZ)")
    st.write(XYZ_ref)

with c2:
    st.subheader("C_a (XYZ)")
    st.write(XYZ_a)

with c3:
    st.subheader("C_b (XYZ)")
    st.write(XYZ_b)

# =====================================================
# STEP 2 — DISTANCE IN XYZ
# =====================================================
st.divider()
st.header("📏 Step 2 — Euclidean Distance in XYZ")

def dist(a, b):
    return np.linalg.norm(a - b)

d_xyz_a = dist(XYZ_ref, XYZ_a)
d_xyz_b = dist(XYZ_ref, XYZ_b)

st.latex("d_{XYZ}(C_{ref}, C_a) = " + f"{d_xyz_a:.3f}")
st.latex("d_{XYZ}(C_{ref}, C_b) = " + f"{d_xyz_b:.3f}")

if d_xyz_a < d_xyz_b:
    st.success("✅ In XYZ space, **C_a is closer to C_ref**.")
else:
    st.success("✅ In XYZ space, **C_b is closer to C_ref**.")

# =====================================================
# STEP 3 — XYZ → u′v′
# =====================================================
st.divider()
st.header("🎯 Step 3 — Convert XYZ → u′v′")

def xyz_to_uv(XYZ):
    X, Y, Z = XYZ
    denom = X + 15*Y + 3*Z
    u = 4*X / denom
    v = 9*Y / denom
    return np.array([u, v])

uv_ref = xyz_to_uv(XYZ_ref)
uv_a   = xyz_to_uv(XYZ_a)
uv_b   = xyz_to_uv(XYZ_b)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("C_ref (u′v′)")
    st.write(uv_ref)

with c2:
    st.subheader("C_a (u′v′)")
    st.write(uv_a)

with c3:
    st.subheader("C_b (u′v′)")
    st.write(uv_b)

# =====================================================
# STEP 4 — DISTANCE IN u′v′
# =====================================================
st.divider()
st.header("📏 Step 4 — Euclidean Distance in u′v′")

d_uv_a = dist(uv_ref, uv_a)
d_uv_b = dist(uv_ref, uv_b)

st.latex("d_{u'v'}(C_{ref}, C_a) = " + f"{d_uv_a:.6f}")
st.latex("d_{u'v'}(C_{ref}, C_b) = " + f"{d_uv_b:.6f}")

if d_uv_a < d_uv_b:
    st.success("✅ In u′v′ space, **C_a is closer to C_ref**.")
else:
    st.success("✅ In u′v′ space, **C_b is closer to C_ref**.")

# =====================================================
# STEP 5 — VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Step 5 — Chromaticity Visualization")

fig, ax = plt.subplots()

ax.scatter(uv_ref[0], uv_ref[1], s=120, label="C_ref")
ax.scatter(uv_a[0], uv_a[1],   s=120, label="C_a")
ax.scatter(uv_b[0], uv_b[1],   s=120, label="C_b")

ax.plot([uv_ref[0], uv_a[0]], [uv_ref[1], uv_a[1]], "--", alpha=0.5)
ax.plot([uv_ref[0], uv_b[0]], [uv_ref[1], uv_b[1]], "--", alpha=0.5)

ax.set_xlabel("u′")
ax.set_ylabel("v′")
ax.set_title("CIE u′v′ Chromaticity")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# =====================================================
# STEP 6 — PERCEPTUAL DISCUSSION
# =====================================================
st.divider()
st.header("🧠 (c) Perceptual Similarity")

st.markdown("""
- **XYZ space is not perceptually uniform**  
  → Euclidean distances do NOT correspond well to human perception.

- **CIE u′v′ space is closer to perceptual uniformity**  
  → Distances better reflect visual similarity.

### ✅ Therefore
The result from the **u′v′ distance** should be trusted more when deciding perceptual similarity.

In practice, even better spaces include:
- CIELAB (ΔE)
- CIECAM02
""")

st.caption("🚀 Fully reproducible, step-by-step color distance analysis.")
