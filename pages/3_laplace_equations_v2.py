import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_radiating_source(domain_size=50, center=(25, 25)):
    """Plot a radiating source with potential decaying outward."""
    x = np.linspace(0, domain_size, 100)
    y = np.linspace(0, domain_size, 100)
    X, Y = np.meshgrid(x, y)

    # Center of the source
    cx, cy = center

    # Compute distance from each point to the center
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Potential decays as 1/R (inverse square law)
    phi = 10 / R  # Simplified for visualization; adjust if needed

    # Clamp values near boundaries to avoid NaN
    phi[R < 1] = 10  # Avoid division by zero at center
    phi[phi == np.inf] = 0  # Handle infinity at edges

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, phi, levels=20)
    st.pyplot(fig)

st.title("Radiating Source Potential Visualization")
domain_size = st.slider("Domain size (10-50)", 10, 50, 30)
center_x_range = (-domain_size // 2 + 10, domain_size - 10)  # Adjustable range
center_y_range = center_x_range

# Use a tuple for slider input (min, max)
center_x, center_y = st.slider(
    "Center coordinates",
    value=(domain_size // 2, domain_size // 2),
    min_value=center_x_range[0],
    max_value=center_x_range[1]
)

plot_radiating_source(domain_size, (center_x, center_y))
