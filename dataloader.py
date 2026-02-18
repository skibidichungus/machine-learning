import torch
import torch.nn as nn

model = nn.Linear(1, 1)  # single-feature linear model: y = Wx + b

x = torch.tensor([[2.0]])  # input feature (batch size 1)
y = torch.tensor([[5.0]])  # target output

pred = model(x)  # forward pass / prediction
print(pred)  # show model's current guess

