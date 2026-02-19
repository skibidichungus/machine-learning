import torch

# Input and target
x = torch.tensor([2.0])
y = torch.tensor([4.0])

# Weight (parameter we want to learn)
w = torch.tensor([1.0], requires_grad=True)

# Forward pass
y_hat = w * x

# Loss
loss = (y_hat - y) ** 2

print("Prediction:", y_hat.item())
print("Loss:", loss.item())

# Backward pass (autograd computes dL/dw)
loss.backward()

print("Gradient dL/dw:", w.grad.item())
