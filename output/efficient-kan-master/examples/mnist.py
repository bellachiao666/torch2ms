import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from src.efficient_kan import KAN

# Train on MNIST
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
from tqdm import tqdm

# Load MNIST
transform = ms.dataset.transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # 'torchvision.transforms.ToTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.transforms.Normalize' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)  # 'torchvision.datasets.MNIST' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)  # 'torchvision.datasets.MNIST' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)  # 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
valloader = DataLoader(valset, batch_size=64, shuffle=False)  # 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

# Define model
model = KAN([28 * 28, 64, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'torch.cuda.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
model.to(device)
# Define optimizer
optimizer = mint.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  # 'torch.optim.lr_scheduler.ExponentialLR' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

# Define loss
criterion = nn.CrossEntropyLoss()
for epoch in range(10):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    # 'torch.no_grad' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )
