# evaluate.py
# Load a trained model (MLP or CNN), run it on the MNIST test set, and produce:
#   1. overall test accuracy
#   2. a confusion matrix (which digits get confused for which)
#   3. a grid of misclassified examples (predicted vs. true labels)
# Fill in each section yourself; the comments are your roadmap.


# =========================
# 1. Imports
# =========================
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import the model class you want to evaluate
# from cnn_classifier import CNN
# from mnist_classifier import <YourMLPClass>
from cnn_classifier import CNN



# =========================
# 2. Config
# =========================
# choose which model to load ("cnn_mnist.pt" or "mnist.pt")
# set device ("cuda" if available else "cpu")
# set batch_size for evaluation (e.g. 128)
load_model = "cnn_mnist.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64


# =========================
# 3. Load test data
# =========================
# build the same transform used during training
#       (ToTensor + Normalize((0.1307,), (0.3081,)))
# load MNIST test set with train=False
# wrap it in a DataLoader (shuffle=False so indices stay stable)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
test_dataset = datasets.MNIST('./MNIST', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# =========================
# 4. Load the model
# =========================
model = CNN() # instantiate the model class (e.g. model = CNN())
model.load_state_dict(torch.load(load_model, map_location=device))
model.to(device)
model.eval()



# =========================
# 5. Run inference, collect predictions + truths
# =========================
# with torch.no_grad():
#         for images, labels in test_loader:
#             - move images to device
#             - logits = model(images)
#             - preds = logits.argmax(dim=1).cpu()
#             - append preds, labels, images.cpu() to their lists
# concatenate the lists into single tensors
all_preds = []
all_labels = []
all_images = [] # keep these around for the misclassified grid
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()
        all_preds.append(predictions)
        all_labels.append(labels.cpu())
        all_images.append(images.cpu())
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_images = torch.cat(all_images)

# =========================
# 6. Overall accuracy
# =========================
correct = (all_preds == all_labels).sum().item()
total = all_labels.size(0)
print(f"Test accuracy: {100 * correct / total:.2f}%")

# =========================
# 7. Confusion matrix
# =========================
# build a 10x10 integer matrix cm
#       for true, pred in zip(all_labels, all_preds):
#           cm[true, pred] += 1
# plot it with plt.imshow(cm, cmap="Blues")
#       - annotate each cell with its count (plt.text in a double loop)
#       - xlabel "Predicted", ylabel "True"
#       - title "Confusion Matrix"
#       - plt.colorbar()
confusion_matrix = np.zeros((10, 10), dtype=int)
for true, pred in zip(all_labels, all_preds):
    confusion_matrix[true, pred] += 1
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, cmap="Blues")
plt.colorbar()

for i in range(10):
    for j in range(10):
        plt.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# 8. Misclassified examples grid
# =========================
# find indices where all_preds != all_labels
# take the first N (e.g. 25) of them
# fig, axes = plt.subplots(5, 5, figsize=(8, 8))
#       for each subplot:
#           - imshow the image (remember to un-normalize if you want it to look right)
#           - set title f"pred {p} / true {t}"
#           - axis off
wrong_indices = (all_preds != all_labels).nonzero(as_tuple=True)[0]
wrong_indices = wrong_indices[:25]
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for idx, ax in zip(wrong_indices, axes.flatten()):
    img = all_images[idx].squeeze()   # shape: (1, 28, 28) → (28, 28)
    p = all_preds[idx].item()
    t = all_labels[idx].item()
    ax.imshow(img, cmap="gray")
    ax.set_title(f"pred {p} / true {t}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("misclassified.png")
plt.show()


# =========================
# 9. (Optional) Stretch ideas
# =========================
# TODO: accept the model path as a CLI arg so this works for any saved model
# TODO: print per-class accuracy (diagonal / row sums of the confusion matrix)
# TODO: print the top 3 most-confused digit pairs
