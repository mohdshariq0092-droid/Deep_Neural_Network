import numpy as np
import matplotlib.pyplot as plt


def plot_perceptron_case(x, y, w_old, w_new, b=0):

    # (num1, num2) is a Python tuple.
    # Tuples are basic Python data structures. They can hold numbers, strings, etc.
    # But tuples don’t support vectorized operations like addition of vectors, multiplication, or dot products.
    # np.array() converts a Python list or tuple into a NumPy array
    # NumPy arrays are designed for numerical computations

    x = np.array(x)
    w_old = np.array(w_old)
    w_new = np.array(w_new)

    max_val = max(
        np.linalg.norm(w_old),        # np.linalg.norm(x) by default calculates the Euclidean norm, also called the 2-norm
        np.linalg.norm(w_new),        # eg # ||x||^2 = sqrt(2^2 + 3^2)
        np.linalg.norm(x)
    ) + 2


    # np.meshgrid(x, y) takes two 1D arrays:
    # x = np.linspace(-max_val, max_val, 400) → 1D array of x-values
    # y = np.linspace(-max_val, max_val, 400) → 1D array of y-values

    # and produces two 2D arrays (xx and yy) that represent all combinations of x and y coordinates.
    # xx contains the x-coordinate of each grid point repeated along rows.
    # yy contains the y-coordinate of each grid point repeated along columns.

    # This is perfect for evaluating functions like f(x,y) over a 2D grid.

    xx, yy = np.meshgrid(
        np.linspace(-max_val, max_val, 400), # np.linspace(start, stop, num) is a NumPy function that returns num evenly spaced numbers between start and stop (inclusive).
        np.linspace(-max_val, max_val, 400)
    )

    def score(w):
        return w[0]*xx + w[1]*yy + b

    def analyze(w):
        dot = np.dot(w, x)
        pred = 1 if dot >= 0 else -1
        correct = (y * dot) > 0
        return dot, pred, correct

    plt.figure(figsize=(14, 6))

    for i, (w, title, color) in enumerate([
        (w_old, "OLD WEIGHT", "red"),
        (w_new, "NEW WEIGHT", "green")
    ], start=1):

        dot, pred, correct = analyze(w)

        plt.subplot(1, 2, i)

        # Regions (prediction-based)
        plt.contourf(xx, yy, score(w),
                     levels=[-1e9, 0, 1e9],
                     colors=['#ffdddd', '#ddffdd'],
                     alpha=0.6)

        # Decision boundary
        plt.contour(xx, yy, score(w),
                    levels=[0],
                    colors=color,
                    linestyles='--')

        # Data point
        plt.scatter(
            x[0], x[1],
            color='blue',
            s=140,
            edgecolors='black',
            label='Data Point'
        )

        # Weight vector
        plt.quiver(0, 0, w[0], w[1],
                   angles='xy',
                   scale_units='xy',
                   scale=1,
                   color=color,
                   width=0.012)

        status = "CORRECT ✅" if correct else "WRONG ❌"

        plt.title(
            f"{title}\n"
            f"w·x = {dot:.2f},  pred = {pred},  y = {y}\n"
            f"{status}, |w·x| (confidence) = {abs(dot):.2f}"
        )

        plt.axhline(0)
        plt.axvline(0)
        plt.gca().set_aspect('equal')
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
        plt.grid(True)
        plt.legend()

    plt.suptitle(
        "Perceptron: Prediction vs Truth vs Confidence",
        fontsize=14
    )
    plt.show()


# =====================
# YOUR EXAMPLE
# =====================
plot_perceptron_case(
    x=(1, 2),
    y=-1,
    w_old=(2, 3),
    w_new=(1, 1)
)
