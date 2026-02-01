import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x1, x2):
    return 2 * x1**3 - 6 * x1 * x2 + 3 * x2**2

# Generate x1 and x2 values
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
x1, x2 = np.meshgrid(x1, x2)

# Calculate f(x1, x2)
z = f(x1, x2)

# Create the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contour(x1, x2, z, levels=20, cmap='viridis')

# Add labels and title
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour Plot of f(x1, x2)')

# Add color bar
plt.colorbar(contour, label='f(x1, x2)')

# Add origin lines
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # x-axis
plt.axvline(0, color='black', linestyle='--', linewidth=1)  # y-axis

# Show the plot
plt.grid(alpha=0.3)
plt.show()