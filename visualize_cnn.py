import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model definition (must match cnn_classifier.py) ──────────────────────────
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


# ── Load model ────────────────────────────────────────────────────────────────
model = CNN().to(device)
model.load_state_dict(torch.load("cnn_mnist.pt", map_location=device))
model.eval()

# ── Test dataset ──────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST("./MNIST", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Training curves
# ═══════════════════════════════════════════════════════════════════════════════
metrics = torch.load("cnn_metrics.pt", map_location="cpu")
train_losses = metrics["train_losses"]
test_accuracies = metrics["test_accuracies"]
epochs = range(1, len(train_losses) + 1)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Loss", color="tab:blue")
ax1.plot(epochs, train_losses, color="tab:blue", marker="o", label="Train Loss")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Test Accuracy (%)", color="tab:orange")
ax2.plot(epochs, test_accuracies, color="tab:orange", marker="s", label="Test Accuracy")
ax2.tick_params(axis="y", labelcolor="tab:orange")

fig.suptitle("CNN Training Curves")
fig.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()
print("Saved training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════════
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

cm = np.zeros((10, 10), dtype=int)
for true, pred in zip(all_labels, all_preds):
    cm[true][pred] += 1

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))
ax.set_xlabel("Predicted digit")
ax.set_ylabel("True digit")
ax.set_title("Confusion Matrix — MNIST Test Set")

for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=8)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sample predictions (4×8 grid of 32 random test images)
# ═══════════════════════════════════════════════════════════════════════════════
indices = np.random.choice(len(test_dataset), 32, replace=False)
images_raw = []
true_labels = []

for idx in indices:
    img, label = test_dataset[idx]
    images_raw.append(img)
    true_labels.append(label)

batch = torch.stack(images_raw).to(device)
with torch.no_grad():
    pred_labels = model(batch).argmax(dim=1).cpu().numpy()

fig, axes = plt.subplots(4, 8, figsize=(14, 8))
fig.suptitle("Sample Predictions  (green = correct, red = wrong)", fontsize=13)

for ax, img, true, pred in zip(axes.flat, images_raw, true_labels, pred_labels):
    # Unnormalize for display: pixel = tensor * std + mean
    display = img.squeeze().numpy() * 0.3081 + 0.1307
    ax.imshow(display, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    color = "green" if pred == true else "red"
    ax.set_title(f"T:{true} P:{pred}", color=color, fontsize=8)

plt.tight_layout()
plt.savefig("predictions.png", dpi=150)
plt.close()
print("Saved predictions.png")
