import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from __future__ import print_function

import argparse

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from torchvision import datasets, transforms


class Net(msnn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)  # 'torch.nn.LSTM' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def construct(self, input):
        # Shape of input is (batch_size,1, 28, 28)
        # converting shape of input to (batch_size, 28, 28)
        # as required by RNN when batch_first is set True
        input = input.reshape(-1, 28, 28)
        output, hidden = self.rnn(input)

        # RNN output shape is (seq_len, batch, input_size)
        # Get last output of RNN
        output = output[:, -1, :]
        output = self.batchnorm(output)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = nn.functional.relu(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        output = mint.special.log_softmax(output, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 'torch.no_grad' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += ms.Tensor.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.dry_run:
                break

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example using RNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--accel', action='store_true',
                        help='enables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='for Saving the current Model')
    args = parser.parse_args()

    if args.accel:
        device = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    else:
        device = torch.device("cpu")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    torch.manual_seed(args.seed)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.accel else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=ms.dataset.transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)  # 'torchvision.transforms.ToTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.transforms.Normalize' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.datasets.MNIST' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 存在 *args/**kwargs，未转换，需手动确认参数映射;
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=ms.dataset.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)  # 'torchvision.transforms.ToTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.transforms.Normalize' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.datasets.MNIST' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 存在 *args/**kwargs，未转换，需手动确认参数映射;

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  # 'torch.optim.Adadelta' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # 'torch.optim.lr_scheduler.StepLR' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_rnn.pt")  # 'torch.save' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


if __name__ == '__main__':
    main()
