import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available, please ensure you have a GPU and CUDA installed."
    assert torch.cuda.device_count() >= args.gpus, f"Only {torch.cuda.device_count()} GPUs are available."

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=0, world_size=1)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    model = SimpleNet().to(args.gpus)
    model = DDP(model, device_ids=[args.gpus], output_device=args.gpus)
    criterion = nn.CrossEntropyLoss().to(args.gpus)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size * args.gpus, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)

    for epoch in range(2):
        start_time = time.time()
        for data, target in train_loader:
            data, target = data.to(args.gpus), target.to(args.gpus)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        if epoch == 1:
            print(f"GPUs: {args.gpus}, Batch size per GPU: {args.batch_size}, Epoch {epoch + 1} training time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
