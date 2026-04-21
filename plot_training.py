# plot_training.py
# Load the metrics saved by cnn_classifier.py (cnn_metrics.pt) and plot
# training loss + test accuracy across epochs.
# Fill in each section yourself; the comments are your roadmap.


# =========================
# 1. Imports
# =========================
import torch
import matplotlib.pyplot as plt


# =========================
# 2. Load the saved metrics
# =========================
metrics = torch.load("cnn_metrics.pt")
train_losses = metrics["train_losses"]
test_accuracies = metrics["test_accuracies"]
epochs = range(1, len(train_losses) + 1)

# =========================
# 3. Plot training loss
# =========================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epochs, train_losses, marker="o", color="steelblue")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)


# =========================
# 4. Plot test accuracy
# =========================
axes[1].plot(epochs, test_accuracies, marker="o", color="darkorange")
axes[1].set_title("Test Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True)



# =========================
# 5. Show / save the figure
# =========================
plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()


# =========================
# 6. (Optional) Stretch ideas
# =========================
# TODO: accept the metrics filename as a CLI arg so this works for any run
# TODO: overlay curves from multiple runs (e.g. MLP vs CNN) on the same plot
# TODO: print a summary line: best epoch, best accuracy, final loss
