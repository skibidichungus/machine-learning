# (A) imports + seed
import numpy as np
np.random.seed(0)

# (B) data: X, y
x = np.random.randn(200, 1)
y = 2.0 * x - 1.0 + 0.1 * np.random.randn(200, 1)
N, D = x.shape

STEPS = 2000
lr = 0.05
H = 16 # Hidden layer size (how many neurons in hidden layer)

# (C) init sizes + params
def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(x.dtype)



W1 = 0.1 * np.random.rand(D, H) # (1, 16)
b1 = np.zeros((1, H))   # (1, 16)
W2 = 0.1 * np.random.rand(H, 1) # (16, 1)
b2 = np.zeros((1, 1))   # (1, 1)

for step in range(STEPS):
    # (D) forward pass
    z1 = x @ W1 + b1    # Linear... and whats with the @ symbol
    a1 = relu(z1)       # ReLU used here
    y_pred = a1 @ W2 + b2 # Whats with the @ symbol

    # (E) loss
    loss = np.mean((y_pred - y)**2)

    # (F) backward pass (grads)
    dy = (2.0 / N) * (y_pred - y)
    da1 = dy @ W2.T
    dz1 = da1 * relu_grad(z1)

    # compute derivatives for weights and bias
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    dW2 = a1.T @ dy
    db2 = np.sum(dy, axis=0, keepdims=True)

    # (G) update params
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    # (H) print occasionally
    if step % 200 == 0:
        print(f"step={step:4d} loss={loss:.6f}")


# (I) final print
print("final loss:", float(loss))

