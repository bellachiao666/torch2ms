import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import os
# import torch
# import torch.optim as optim


def train(rank, args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)  # 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 存在 *args/**kwargs，未转换，需手动确认参数映射;

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # 'torch.optim.SGD' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)


def test(args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)  # 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 存在 *args/**kwargs，未转换，需手动确认参数映射;

    test_epoch(model, device, test_loader)


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = nn.functional.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 'torch.no_grad' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += nn.functional.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
