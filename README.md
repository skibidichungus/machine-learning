# Machine Learning from Scratch

Learning machine learning fundamentals by building things up from the basics — derivatives, gradient descent, neural networks, and eventually training a real classifier on MNIST.

## What's in here

**Math & foundations**
- `derivative.py` — numerical derivatives and plotting
- `gradient_descent.py` — gradient descent on a simple function
- `chain_rule.py` — autograd and the chain rule
- `matrix_gradient.py` — gradients with matrix multiplication
- `cross_entropy.py` — cross-entropy loss demo
- `activations.py` — plotting ReLU, Sigmoid, and Tanh

**Linear models**
- `numpy_gd.py` — gradient descent from scratch with just NumPy
- `linearmodel.py` — linear regression with PyTorch
- `extra.py` — step-by-step SGD walkthrough
- `dataloader.py` — batching data with DataLoader

**Neural networks**
- `numpy_2layer_relu.py` — 2-layer network with ReLU, built entirely in NumPy
- `mlp_classification.py` — MLP classifier on the Iris dataset
- `mnist_classifier.py` — full MNIST pipeline (train, evaluate, save, load)
- `cnn_classifier.py` — convolutional neural network for MNIST (Conv → ReLU → Pool ×2 → FC)

## Setup

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Then just run any file:

```bash
python mnist_classifier.py
```
