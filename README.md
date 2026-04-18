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
- `cnn_classifier.py` — convolutional neural network for MNIST (Conv → ReLU → Pool ×2 → FC), **99.13% test accuracy**
- `visualize_cnn.py` — visualizations for the trained CNN (training curves, confusion matrix, sample predictions)

## Setup

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Then just run any file:

```bash
python mnist_classifier.py
```

## CNN Visualizations

After training, run the visualization script to generate three plots:

```bash
python cnn_classifier.py   # trains and saves cnn_mnist.pt + cnn_metrics.pt
python visualize_cnn.py    # saves training_curves.png, confusion_matrix.png, predictions.png
```

- `training_curves.png` — per-epoch train loss and test accuracy
- `confusion_matrix.png` — 10×10 heatmap of predicted vs true digits
- `predictions.png` — 32 random test images with true/predicted labels (green = correct, red = wrong)
