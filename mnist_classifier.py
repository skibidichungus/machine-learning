import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# (A) device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (B) load MNIST dataset
transform = transforms.ToTensor()
train_data = datasets.MNIST(root=".", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=".", train=False, download=True, transform=transform)   # MNIST test split
print("MNIST loaded:", len(train_data), "samples")


# (C) create DataLoader (batching)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# (D) define model
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = Net().to(device)    # Model created


# (E) loss + optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    # (F) training loop over batches
        # forward
        # loss
        # backward
        # step
    model.train()   #.train() is PyTorch method

    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)    # forward
        loss = loss_fn(pred, yb)    # loss
        optimizer.zero_grad()   # zero gradient
        loss.backward()     # backward/compute gradients
        optimizer.step()    # update weights

        if (batch_idx + 1) % 200 == 0:
            print("batch", batch_idx + 1, "loss", loss.item())

    print("epoch done")

# Evaluation loop that computes accuracy
model.eval()    # Switches off training

correct = 0
total = 0

with torch.no_grad():   # saves memory + speed
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        preds = logits.argmax(dim=1)    # picks the predicted class 0-9

        correct += (preds == yb).sum().item()
        total += yb.size(0)
acc = correct / total
print(f"Test accuracy: {acc:4f}")

# (G) save model
torch.save(model.state_dict(), "mnist.pt")
print("saved model!")

# (H) load model into a fresh instance + re-evaluate
loaded_model = Net().to(device)
loaded_model.load_state_dict(torch.load("mnist.pt", map_location=device))
loaded_model.eval()

correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = loaded_model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print(f"loaded model test accuracy: {correct/total:.4f}")
