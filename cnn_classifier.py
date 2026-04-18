# cnn_classifier.py
# A convolutional neural network for MNIST — the next step up from mnist_classifier.py.
# Fill in each section yourself; the comments are your roadmap.


# =========================
# 1. Imports
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms




# =========================
# 2. Hyperparameters / config
# =========================
batch_size = 64
num_epochs = 5
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 3. Data loading
# =========================
# Build a transform pipeline
#       - transforms.ToTensor()
#       - transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean/std
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# load the training set from ./MNIST with train=True, download=True, transform=transform
train_dataset = datasets.MNIST("./MNIST", train=True, download=True, transform=transform)

# load the test set with train=False
test_dataset = datasets.MNIST("./MNIST", train=False, download=True, transform=transform)

# wrap both in DataLoader (shuffle=True for train, shuffle=False for test)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# =========================
# 4. Model definition
# =========================
# define a class CNN(nn.Module)
#       Suggested architecture:
#         - conv1: Conv2d(1, 32, kernel_size=3, padding=1)
#         - conv2: Conv2d(32, 64, kernel_size=3, padding=1)
#         - pool:  MaxPool2d(2, 2)
#         - fc1:   Linear(64 * 7 * 7, 128)
#         - fc2:   Linear(128, 10)
#       forward pass:
#         - conv1 -> ReLU -> pool          (28x28 -> 14x14)
#         - conv2 -> ReLU -> pool          (14x14 -> 7x7)
#         - flatten
#         - fc1 -> ReLU
#         - fc2   (logits, no softmax — CrossEntropyLoss applies it)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))

        return self.fc2(x)


# =========================
# 5. Instantiate model, loss, optimizer
# =========================
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# =========================
# 6. Training loop
# =========================
# TODO: for epoch in range(num_epochs):
#         model.train()
#         for images, labels in train_loader:
#             - move to device
#             - zero gradients
#             - forward pass -> logits
#             - compute loss
#             - loss.backward()
#             - optimizer.step()
#         print average training loss for the epoch
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")



# =========================
# 7. Evaluation
# =========================
# TODO: model.eval()
# TODO: with torch.no_grad():
#         - iterate over test_loader
#         - predictions = logits.argmax(dim=1)
#         - accumulate correct / total
# TODO: print test accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
print(f"Test accuracy: {100 * correct / total:.2f}%")

# =========================
# 8. Save the trained model
# =========================
torch.save(model.state_dict(), "cnn_mnist.pt")


# =========================
# 9. (Optional) Compare against your MLP
# =========================
# TODO: note your MLP test accuracy from mnist_classifier.py
# TODO: note CNN test accuracy — should see a meaningful improvement


# =========================
# 10. (Optional) Stretch ideas once it works
# =========================
# TODO: visualize a few misclassified test images with matplotlib
# TODO: add dropout after fc1 and see if test accuracy improves
# TODO: plot training loss per epoch
