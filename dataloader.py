import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

x = torch.randn(100, 1)
y = 2 * x + 0.5

ds = TensorDataset(x, y)
dl = DataLoader(ds, batch_size=16, shuffle=True)

model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(10):
    for xb, yb in dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()
