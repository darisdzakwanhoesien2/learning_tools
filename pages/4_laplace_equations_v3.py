import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_radiating_source(
    domain_size=50,
    center=(25, 25),
    shape_ratio=0.5  # Transition from square (0) to circle (1)
):
    """Plot a radiating source with potential decaying outward in a circular/square domain."""
    x = np.linspace(0, domain_size, 100)
    y = np.linspace(0, domain_size, 100)
    X, Y = np.meshgrid(x, y)

    # Center of the source
    cx, cy = center

    # Compute distance from each point to the center
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Adjust shape based on ratio: 0=square, 1=circle
    if shape_ratio == 0:
        # Square domain: r/2 = 0 → only points inside a square
        mask = ((abs(X - cx) <= domain_size / 2) & (abs(Y - cy) <= domain_size / 2))
        R[~mask] = np.inf  # Set outer boundary to infinity

    elif shape_ratio == 1:
        # Circle domain: r/2 = 1 → only points inside a circle
        mask = (R <= domain_size / 2)
        R[~mask] = np.inf  # Set outer boundary to infinity

    else:
        # Linear interpolation between square and circle
        # For shape_ratio in [0,1], use a weighted combination
        mask_square = ((abs(X - cx) <= (domain_size / 2) * (1 - shape_ratio)) &
                      (abs(Y - cy) <= (domain_size / 2) * (1 - shape_ratio)))
        mask_circle = (R <= (domain_size / 2) * (1 + shape_ratio))
        R[~mask_square & ~mask_circle] = np.inf

    # Potential decays as 1/R (inverse square law)
    phi = 10 / R
    phi[phi == np.inf] = 0  # Avoid NaN at boundaries

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, phi, levels=20, cmap="viridis")
    st.pyplot(fig)

st.title("Laplace Equation: Radiating Source in Shaped Domain")
domain_size = st.slider("Domain size (10-50)", 10, 50, 30)
center_x, center_y = st.slider(
    "Center coordinates",
    value=(domain_size // 2, domain_size // 2),
    min_value=0,
    max_value=domain_size
)

# Slider for shape transition (square=0 → circle=1)
shape_ratio = st.slider("Shape ratio (0=Square, 1=Circle)", 0.0, 1.0, 0.5)

plot_radiating_source(domain_size, (center_x, center_y), shape_ratio)
