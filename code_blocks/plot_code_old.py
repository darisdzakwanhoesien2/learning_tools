
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
x1, x2 = np.meshgrid(x1, x2)

z = f(x1, x2)  # Assuming 'f' is defined elsewhere

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='k', alpha=0.8)
plt.show()
