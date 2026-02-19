import matplotlib.pyplot as plt

# Function: f(x) = x^2
def f(x):
    return x**2

# Derivative: f'(x) = 2x
def f_prime(x):
    return 2 * x

# Gradient descent settings
x = 5.0 # Starting position
lr = 1.1 # Learning rate (step size)
steps = 20 # How many updates

history = [] # Empty array to hold gradient(derivate updates)

print("Running gradient descent...\n")

for step in range(steps):

    history.append(x)

    # Compute gradient at x?
    grad = f_prime(x)

    # Update rule
    x = x - lr * grad

    print(f'Step {step:2d}: x = {x:.6f}, f(x) = {f(x):.6f}')


# Plot how x changes over time
plt.plot(history)
plt.title("Gradient Descent: x moving toward 0")
plt.xlabel("Step")
plt.ylabel("x value")
plt.show()