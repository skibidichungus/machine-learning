import torch
import torch.nn as nn

# data
torch.manual_seed(0) # same numbers every run
x = torch.randn(100, 1) # 100 samples, 1 feature each #1 Tensor/array
true_w, true_b = 2.0, -1.0  # add real slope and y intercept, the model doesnt know this
y = true_w * x + true_b + 0.1 * torch.randn(100, 1) # no noise, to add noise add + 0.1 * torch.randn(100, 1)
# print("x shape:", x.shape)
# print("y shape:", y.shape)

model = nn.Linear(1, 1)                 # y = Wx + b, #2 Model
loss_fn = nn.MSELoss()  # Mean squared error
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(200):
    y_pred = model(x)   # Forward pass/prediction
    loss = loss_fn(y_pred, y)   # Loss/How wrong i am

    opt.zero_grad()
    loss.backward() # Compute gradients
    opt.step()  # Optimizer step/update weights

    if step % 50 == 0:
        print(step, loss.item())


print("W,b:", model.weight.item(), model.bias.item())
