import torch
import torch.nn.functional as F

# Logits (raw model output, NOT probabilities)
logits = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)

# Correct class index
target = torch.tensor([0])  # Class 0 is correct

# Cross entropy loss
loss = F.cross_entropy(logits, target)

print("Loss: ", loss.item())

loss.backward()

print("\nGradient w.r.t logits")
print(logits.grad)