import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from __future__ import print_function
import argparse
# import torch
# import torch.nn as nn
# import torch.multiprocessing as mp
# from torchvision import datasets, transforms

from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--mps', action='store_true', default=False,
                    help='enables macOS GPU training')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save the trained model to state_dict')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')

class Net(msnn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def construct(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training = self.training)
        x = self.fc2(x)
        return mint.special.log_softmax(x, dim=1)


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()  # 'torch.cuda.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    use_mps = args.mps and torch.backends.mps.is_available()  # 'torch.backends.mps.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    if use_cuda:
        device = torch.device("cuda")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    elif use_mps:
        device = torch.device("mps")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    else:
        device = torch.device("cpu")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    transform=ms.dataset.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])  # 'torchvision.transforms.ToTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.transforms.Normalize' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)  # 'torchvision.datasets.MNIST' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)  # 'torchvision.datasets.MNIST' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })

    torch.manual_seed(args.seed)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    mp.set_start_method('spawn', force=True)  # 'torch.multiprocessing.set_start_method' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    model = Net().to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataset1, kwargs))  # 'torch.multiprocessing.Process' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if args.save_model:
        torch.save(model.state_dict(), "MNIST_hogwild.pt")  # 'torch.save' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    # Once training is complete, we can test the model
    test(args, model, device, dataset2, kwargs)
