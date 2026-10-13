import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(train_data, batch_size=128, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.mean = nn.Linear(64, 2)
        self.logvar = nn.Linear(64, 2)
        self.fc3 = nn.Linear(2, 64)
        self.fc4 = nn.Linear(64, 256)
        self.fc5 = nn.Linear(256, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mean(h), self.logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar


def vae_loss(recon, x, mean, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return bce + kl


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(15):
    total = 0
    for x, _ in loader:
        x = x.view(-1, 784).to(device)
        optimizer.zero_grad()
        recon, mean, logvar = model(x)
        loss = vae_loss(recon, x, mean, logvar)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print("epoch", epoch + 1, "loss", total / len(loader.dataset))

model.eval()
with torch.no_grad():
    z = torch.randn(16, 2).to(device)
    samples = model.decode(z).view(16, 28, 28).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, img in zip(axes.flat, samples):
    ax.imshow(img, cmap="gray")
    ax.axis("off")
plt.show()
