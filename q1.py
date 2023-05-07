import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Prepare the dataset and data loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to measure training time for different batch sizes
def measure_training_time(batch_size):
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    
    for epoch in range(2):
        start_time = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        if epoch == 1:
            print(f"Batch size: {batch_size}, Epoch {epoch + 1} training time: {end_time - start_time:.2f} seconds")

# Test for different batch sizes
batch_sizes = [32, 128, 512]
for batch_size in batch_sizes:
    try:
        measure_training_time(batch_size)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Batch size {batch_size} cannot fit in GPU memory")
            break
        else:
            raise e
