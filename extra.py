import torch
import torch.nn as nn

# Defined a model
model = nn.Linear(1, 1)  # single-feature linear model: y = Wx + b

# Created data
x = torch.tensor([[2.0]])  # input feature (batch size 1)
y = torch.tensor([[5.0]])  # target output

loss_fn = nn.MSELoss() # ONLY defined a loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # ONLY defined optimizer

for i in range(10):
    pred = model(x) # Got prediction
    print("Prediction:",pred.item())

    loss = loss_fn(pred, y) # How wrong the prediction
    

    optimizer.zero_grad()   # Zero'd the gradient
    loss.backward()     # Compute Gradient
    print("Gradient:", model.weight.grad)   # Temp line
    optimizer.step()    # Update weight

    new_pred = model(x)
    print("New Prediction:",new_pred.item())
    new_loss = loss_fn(new_pred, y)
    print("New loss", new_loss.item())

    print(f"Step {i+1}, W={model.weight.item():.4f}, b={model.bias.item():.4f}, pred={pred.item():.4f}, loss={loss.item():.4f}")
    print()
    
