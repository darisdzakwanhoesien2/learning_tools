import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Set page config
st.set_page_config(page_title="Machine Vision Solver", layout="wide")

def solve_ex1_q1():
    st.title("Exercise 1: Pinhole Camera Projection")
    
    st.markdown("""
    The perspective projection equations for a pinhole camera are defined as:
    """)
    st.latex(r"x_{n}=f\frac{x_{c}}{z_{c}}, \quad y_{n}=f\frac{y_{c}}{z_{c}}")
    st.write("where $f$ is the focal length and $[x_c, y_c, z_c]^\top$ is the point in the camera coordinate frame.")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        f = st.number_input("Focal Length (f)", value=1.0)
        xc = st.number_input("Point Xc", value=10.0)
        yc = st.number_input("Point Yc", value=5.0)
        zc = st.number_input("Point Zc (Depth)", value=50.0)

    # Calculation
    if zc != 0:
        xn = f * (xc / zc)
        yn = f * (yc / zc)
        st.success(f"Normalized Image Coordinates: xn = {xn:.4f}, yn = {yn:.4f}")
    else:
        st.error("zc cannot be zero.")

    st.subheader("Geometric Reasoning")
    st.write("""
    **Reasoning:** These equations are derived from **similar triangles** formed by the 3D point, the optical center (pinhole), and the image plane. 
    
    **Virtual Image:** If we assume a virtual image plane is located at distance $f$ **in front** of the pinhole, the equations for $x_n$ and $y_n$ remain mathematically the same, but the image is no longer inverted (no sign change required).
    """)

def solve_ex5_q1():
    st.title("Exercise 5: Nearest Class Mean & KNN")
    
    # Data from Source
    c1 = np.array([, 
                  ])
    c2 = np.array([, 
                  ]) # Note: y row based on available source text
    
    test_point = np.array([9.6, 2])
    
    st.write("### Dataset")
    st.write("**Class 1 Mean (Centroid):**")
    mean1 = np.mean(c1, axis=1)
    st.write(mean1)
    
    st.write("**Class 2 Mean (Centroid):**")
    mean2 = np.mean(c2, axis=1)
    st.write(mean2)

    # 1(a) Nearest Mean
    dist_m1 = distance.euclidean(test_point, mean1)
    dist_m2 = distance.euclidean(test_point, mean2)
    
    st.subheader("Results")
    st.write(f"Distance to C1 Mean: {dist_m1:.4f}")
    st.write(f"Distance to C2 Mean: {dist_m2:.4f}")
    
    winner_mean = "Class 1" if dist_m1 < dist_m2 else "Class 2"
    st.info(f"**Nearest Mean Classifier Result:** {winner_mean}")

    # 1(c) K-Nearest Neighbors (K=3)
    st.subheader("K-Nearest Neighbors (K=3)")
    all_points = np.hstack((c1, c2)).T
    all_labels = ['C1']*c1.shape + ['C2']*c2.shape
    
    dists = [distance.euclidean(test_point, p) for p in all_points]
    idx = np.argsort(dists)[:3] # Get top 3
    neighbors = [all_labels[i] for i in idx]
    
    st.write(f"Three Nearest Neighbors: {neighbors}")
    st.info(f"**KNN (K=3) Result:** {max(set(neighbors), key=neighbors.count)}")

    # Visualization
    fig, ax = plt.subplots()
    ax.scatter(c1, c1, color='blue', label='Class 1')
    ax.scatter(c2, c2, color='red', label='Class 2')
    ax.scatter(test_point, test_point, color='green', marker='X', s=100, label='Test Point')
    ax.legend()
    st.pyplot(fig)

# Navigation
page = st.sidebar.selectbox("Select Question", ["Ex 1: Pinhole Camera", "Ex 5: KNN Classification"])

if page == "Ex 1: Pinhole Camera":
    solve_ex1_q1()
else:
    solve_ex5_q1()