import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_net import Attention_net 
from data_loader import load_image_features
from tensorboardX import SummaryWriter

data_dir = 'data'
# Load data
print("Reading QA DATA")
qa_data = data_loader.load_questions_answers(2, data_dir)
print("Reading Image DATA")
train_image_features ,train_image_id_list = load_image_features(data_dir, 'train')
val_image_features, val_image_id_list = load_image_features(data_dir, 'val')

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()train