import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Connected Components Labeling", layout="wide")
st.title("🔗 Connected Components Labeling — Union–Find (4-Connectivity)")

# =====================================================
# INPUT IMAGE (FROM PROBLEM)
# =====================================================

B = np.array([
    [1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,1],
    [1,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0],
    [1,1,1,0,1,1,1,1],
    [0,0,1,0,1,0,0,0],
    [1,1,1,0,1,0,1,1],
    [1,1,1,0,1,1,1,1],
], dtype=np.uint8)

st.header("📥 Input Binary Image")

st.dataframe(B)

# =====================================================
# UNION FIND DATA STRUCTURE
# =====================================================

class UnionFind:
    def __init__(self):
        self.parent = {}

    def make_set(self, x):
        self.parent[x] = x

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

# =====================================================
# FIRST PASS
# =====================================================

st.divider()
st.header("🧮 First Pass — Temporary Labels + Equivalences")

h, w = B.shape
labels = np.zeros_like(B, dtype=int)
uf = UnionFind()
next_label = 1

for i in range(h):
    for j in range(w):
        if B[i,j] == 0:
            continue

        neighbors = []

        # 4-connectivity: left and up
        if j > 0 and labels[i, j-1] > 0:
            neighbors.append(labels[i, j-1])
        if i > 0 and labels[i-1, j] > 0:
            neighbors.append(labels[i-1, j])

        if not neighbors:
            labels[i,j] = next_label
            uf.make_set(next_label)
            next_label += 1
        else:
            min_label = min(neighbors)
            labels[i,j] = min_label
            for lab in neighbors:
                uf.union(min_label, lab)

st.write("Temporary label image:")
st.dataframe(labels)

st.subheader("Equivalence Table (Union–Find Parent Map)")
st.write(uf.parent)

# =====================================================
# SECOND PASS
# =====================================================

st.divider()
st.header("🧮 Second Pass — Resolve Final Labels")

final_labels = labels.copy()
label_map = {}
new_label = 1

for i in range(h):
    for j in range(w):
        if labels[i,j] > 0:
            root = uf.find(labels[i,j])

            if root not in label_map:
                label_map[root] = new_label
                new_label += 1

            final_labels[i,j] = label_map[root]

st.write("Final labeled image:")
st.dataframe(final_labels)

num_components = len(label_map)
st.success(f"✅ Number of connected components = {num_components}")

# =====================================================
# VISUALIZATION
# =====================================================

st.divider()
st.header("📊 Visualization")

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(final_labels, cmap="tab20")
ax.set_title("Connected Components")
ax.axis("off")
plt.colorbar(im, ax=ax)

st.pyplot(fig)

# =====================================================
# ALGORITHM EXPLANATION
# =====================================================

st.divider()
st.header("📘 Algorithm Explanation")

st.markdown("""
### First Pass
For each foreground pixel:

1. Look at **left and upper neighbors** (4-connectivity).
2. If no labeled neighbors → assign a new label.
3. If neighbors exist → assign the smallest label.
4. Record equivalences using Union–Find.

### Second Pass
1. Replace each label by its root representative.
2. Relabel compactly: 1, 2, 3, ...

### Advantages
- Efficient: nearly linear time.
- Handles complex equivalences.
- Standard algorithm in image processing.

### Connectivity
Using **4-connectivity** ensures diagonal pixels are NOT connected.
""")

st.caption("🚀 Classical two-pass connected component labeling.")
