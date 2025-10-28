"""
MNIST Training Demo - Getting Started with PyTorch

This is a simple getting started example that demonstrates training and
evaluating a neural network on the MNIST dataset.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch_max_backend import max_backend

os.environ["TORCH_MAX_BACKEND_PROFILE"] = "0"
os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"


class SimpleNet(nn.Module):
    """Simple feedforward neural network for MNIST classification."""

    def __init__(self):
        super().__init__()
        # Input: 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes (digits 0-9)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 784)

        # Hidden layers with RELU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        return x


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        # print(data.shape)
        print("forward pass of batch", batch_idx, "in epoch", epoch)
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        print("calling backward")
        loss.backward()
        print("calling step")
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Print progress
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """Evaluate the model on test data."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            print("calling model eval")
            output = model(data)
            print("done forward eval")
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    # Hyperparameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    learning_rate = 0.01
    momentum = 0.9

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    print("\nLoading MNIST dataset...")
    data_dir = Path.home() / ".pytorch" / "datasets" / "mnist"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model, loss, and optimizer
    model = SimpleNet().to(device)
    print()

    # Compile the model with max_backend
    model = torch.compile(model, backend=max_backend, fullgraph=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print("\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        print("-" * 60)

    print("\nTraining complete!")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Save the model
    model_path = Path(__file__).parent / "mnist_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
