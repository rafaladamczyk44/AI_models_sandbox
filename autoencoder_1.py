import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.Linear(36, 9),
            nn.BatchNorm1d(9),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = AutoEncoder().to(device)
model.apply(weights_init)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

epochs = 5
outputs = []
losses = []

for epoch in range(epochs):
    model.train()
    for image, _ in loader:
        image = image.view(-1, 28 * 28).to(device)
        reconstructed = model(image)
        loss = criterion(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    scheduler.step()
    outputs.append((epoch, image, reconstructed))
    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(losses[-100:])
plt.show()

# Select a few random images for comparison
num_images = 5
random_indices = random.sample(range(len(dataset)), num_images)

fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

model.eval()
with torch.no_grad():
    for i, idx in enumerate(random_indices):
        original_image, _ = dataset[idx]
        original_image = original_image.view(-1, 28 * 28).to(device)
        decoded_image = model(original_image).detach().cpu().view(28, 28)

        axes[i, 0].imshow(original_image.cpu().view(28, 28), cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(decoded_image, cmap='gray')
        axes[i, 1].set_title('Decoded Image')
        axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
