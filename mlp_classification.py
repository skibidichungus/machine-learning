import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data   # Shape (150, 4), 150 flowers and 4 features per sample
y = iris.target # Shape(150)

# Print out raw features or place after A = scalar... for plot after scaling
#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
#plt.title("Iris Dataset (first 2 features)")
#plt.show()

# Print raw data
print("First 5 samples:\n", X[:5])
print("\nFirst 5 labels:\n", y[:5])
print("\nClass names:", iris.target_names)
print("\nFeature names:", iris.feature_names)
print()

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Print out plot after scaling or place after y = iris.target for plot with raw features
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Iris Dataset (first 2 features)")
plt.show()

# Train/test split: Splits dataset into training inputs, test inputs, training labels and test labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)       # test_size means 20% testing, 80% training
# random_state=42 makes it reproducible, when removed it randomly shuffles differently each run, training/tests sets change each time

# Print out here to see training data

# See what data is training on
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nFirst training sample:", X_train[0])
print("First training label:", y_train[0])


# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Define small MLP
model = nn.Sequential(
    nn.Linear(4, 10),   # Input size 4 features, output size 10 neurons
    nn.ReLU(),
    nn.Linear(10,3)     # 10 input, 3 output classes
)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training Loop
for epoch in range(100):
    optimizer.zero_grad()       # Clear old gradients

    logits = model(X_train)     # Forward pass/prediction
    loss = criterion(logits, y_train)   # Compute loss using Cross Entropy

    loss.backward()         # Compute gradients
    optimizer.step()        # Update weights

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Evaluate
with torch.no_grad():
    test_logits = model(X_test)
    predictions = torch.argmax(test_logits, dim=1)      # Picks class with highest logits
    accuracy = (predictions == y_test).float().mean()       # Compares predictions, Converts to float, and takes the mean

print("\nTest Accuracy:", accuracy.item())