# (A) imports + seed
import numpy as np

# (B) make data x, y
STEPS = 500
np.random.seed(0)
x = np.random.randn(100, 1)
true_w = 2.0
true_b = -1.0
y = true_w * x + true_b + 0.1 * np.random.randn(100, 1)

# (C) initialize w, b, lr
w = float(np.random.randn(1, 1))
b = float(np.random.randn(1, 1))
lr = 0.01


for step in range(STEPS):
    # (D) forward: y_pred
    y_pred = w * x + b
    
    # (E) loss: MSE
    loss = np.mean((y_pred - y)**2)
    
    # (F) gradients: dw, db
    err = y_pred - y
    dw = 2.0 * np.mean(err * x)
    db = 2.0 * np.mean(err)
    
    # (G) update: w, b
    w = w - lr * dw
    b = b - lr * db
    
    # (H) print occasionally
    if step % 50 == 0:
        print(f"step={step:4d}  loss={loss:.6f}  w={float(w):.4f}  b={float(b):.4f}")



# (I) final print
print("final:", "w=", w, "b=", b, "loss=", loss)