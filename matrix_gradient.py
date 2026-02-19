import torch

# Input vector (1 sample, 2 features)
x = torch.tensor([[1.0, 2.0]])  # shape (1, 2)

# Target output
y = torch.tensor([[4.0]])   # shape (1, 1)

# Weight matrix (2 weights)
w = torch.tensor([[1.0],    # top is weight 1
                  [1.0]], requires_grad=True)   # shape(2, 1), bottom is weight 2

# Forward pass: prediction
y_hat = x @ w    # Matrix multiplication, shape (1, 1)

# Loss: squared error
loss = (y_hat - y)**2

print("Predication y_hat: ", y_hat.item())
print("Loss: ", loss.item())

# Backward pass
loss.backward()

print("\nGradient dL/dw:")
print(w.grad)