import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchvision import datasets, transforms

# Set the device to mps
device = 'mps'

# LOAD DATA
training_data = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


# Model
# TODO: Expand the model - this model is from an example, try to build different
class Net(nn.Module):
    # Model structure
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    # Model's forward pass
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = Net().to(device)
print(model)


# Optimizer
learning_rate = 0.1

# TODO: Choose optimizer
optim_SGD = SGD(model.parameters(), lr=learning_rate)
optim_adam = Adam(model.parameters(), lr=learning_rate)

# Loss function
loss_cross_entropy = nn.CrossEntropyLoss()


# Training
def model_training(model, train_loader, loss_fn, optimizer):
    # Initialize model training
    model.train()

    # The enumerate() function is one of the built-in functions in Python.
    # It provides a handy way to access each item in an iterable,
    # along with a count value that specifies the order in which the item was accessed

    # Backpropagation
    for batch_idx, (data, target) in enumerate(train_loader):
        # Data = X, target = y
        data, target = data.to(device), target.to(device)

        # Predicted value
        output = model(data)

        # Calculate loss function
        loss = loss_fn(output, target)

        # Initialize backprop
        loss.backward()

        # Optimizer part
        optimizer.step()

        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(data)
            print(f'Loss: {loss}')


loss_history = []

for epoch in range(10):
    print(f'Epoch {epoch}')
    model_training(model, train_loader, loss_fn=loss_cross_entropy, optimizer=optim_SGD)
