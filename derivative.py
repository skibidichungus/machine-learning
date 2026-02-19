import numpy as np
import matplotlib.pyplot as plt

# Function: f(x) = x^2
def f(x):
    return x**2

def derivative_approx(x, h = 1e-5):
    return (f(x + h) - f(x)) / h    # Another way to get derivatives, h need to be tiny 0.0001

# Points
x_vals = np.linspace(-3, 3, 100) # means values between -3 and 3 and 100 points betwen
# Compute f(x) and f'(x)
y_vals = f(x_vals)
dy_vals = derivative_approx(x_vals)

# Plot what are these functions and parameters do i need to have them memorized?
plt.plot(x_vals, y_vals, label="f(x) = x^2")
plt.plot(x_vals, dy_vals, label="Approx f'(x)", linestyle="--")
plt.legend()
plt.title("Function vs Derivative") # Self explanatory dont explain
plt.show()      # Self explanatory dont explain