import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 100)

relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)

plt.plot(x, relu, label="ReLU")
plt.plot(x, sigmoid, label="Sigmoid")
plt.plot(x, tanh, label="Tanh")

plt.legend()
plt.title("Activation Functions")
plt.show()