import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()


def update(frame):
    ax.clear()

    # Mountain slowly grows
    Z = np.sin(np.sqrt(X ** 2 + Y ** 2) + frame * 0.2)

    contour = ax.contourf(X, Y, Z, levels=20, cmap='terrain')
    ax.set_title("Contour Plot = Mountain Map 🏔️")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


ani = FuncAnimation(fig, update, frames=30, interval=200)
plt.show()
