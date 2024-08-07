import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Already transformed and engineered dataset
df = pd.read_csv('data/datasets/diabetic_data_transformed.csv')
train_dataset, test_dataset = train_test_split(df, test_size=0.2)

device = 'cpu'


# Dataset class to read the file
class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df.to_numpy()
        self.X = self.data[:, :-1]
        self.y = self.data[:, -1].astype(int)

        # Change to float32 tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        # Change to int tensor
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self, features_in, features_out):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(features_in, 24)
        self.lin1 = nn.Linear(24, 36)
        self.lin2 = nn.Linear(36, 48)
        self.lin3 = nn.Linear(48, 24)
        self.out = nn.Linear(24, features_out)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.relu(self.lin3(x))

        x = self.tanh(self.out(x))
        return x


# Parameters for model
batch_size = 32
epochs = 20

input_features = len(df.columns) - 1
output_features = df[df.columns[-1]].nunique()

# Load train and test
train_dataset = MyDataset(train_dataset)
test_dataset = MyDataset(test_dataset)
# Load dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Create model, optimizer and loss function
model = Model(features_in=input_features, features_out=output_features).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

losses = []

for epoch in range(epochs):
    running_loss = 0.0
    model.train()

    for batch_id, data in enumerate(train_loader):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(inputs)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_id % 100 == 0:
            avg_loss = running_loss / (batch_id + 1)
            losses.append(loss.item())
            running_loss = 0.0
            print(f'Epoch: {epoch + 1}/{epochs}')
            print(f'Batch: {batch_id}/{len(train_loader)}')
            print(f'Loss: {avg_loss}')
        # inputs, label = inputs.to(device), label.to(device)


model.eval()
correct, total = 0, 0

with torch.no_grad():
    for data in test_loader:
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)
        pred = model(inputs)

        _, predicted = torch.max(pred.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f'Accuracy: {100*correct/total}')
# Plotting the loss values

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Batch count')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()
