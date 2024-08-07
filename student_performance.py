import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# TODO: Current accuracy is around 90%, try to optimize
# TODO: Add data processing

# Load the dataset
df = pd.read_csv('data/datasets/Student_performance_data _.csv')

# Data processing
df['GradeClass'] = df['GradeClass'].astype(int)

# Drop student id column
df.drop(columns='StudentID', axis=1, inplace=True)

# Scaling the data for certain columns
scaler = MinMaxScaler()
columns_to_transform = ['Age', 'StudyTimeWeekly', 'Absences']
df[columns_to_transform] = scaler.fit_transform(df[columns_to_transform])

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Device configuration
device = 'cpu'


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
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_param_1)
        self.fc2 = nn.Linear(hidden_layer_param_1, hidden_layer_param_2)
        self.fc3 = nn.Linear(hidden_layer_param_2, hidden_layer_param_3)
        self.out = nn.Linear(hidden_layer_param_3, output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.45)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.out(x)
        return x


# Parameters
input_size = len(df.columns) - 1
output_size = 5

hidden_layer_param_1 = 24
hidden_layer_param_2 = 48
hidden_layer_param_3 = 64

# Batch
batch_size = 64

# Learning rate
learning_rate = 0.001

# Epochs
epochs = 500

# Create datasets and data loaders
train_dataset = MyDataset(train_df)
test_dataset = MyDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = Model(input_size=input_size, output_size=output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(params=model.parameters(), lr=learning_rate)

losses = []  # List to store loss values


# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    # Set the model to training mode
    model.train()
    for batch_id, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_id % 100 == 0:
            average_loss = running_loss / (batch_id + 1)
            print(f'[{epoch + 1}] loss: {average_loss:.3f}')
            losses.append(average_loss)
            running_loss = 0.0


# Testing the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No need to calculate gradients during testing
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

# Plotting the loss values
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Batch count')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

