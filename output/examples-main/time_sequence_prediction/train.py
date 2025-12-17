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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Sequence(msnn.Cell):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)  # 'torch.nn.LSTMCell' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.lstm2 = nn.LSTMCell(51, 51)  # 'torch.nn.LSTMCell' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.linear = nn.Linear(51, 1)

    def construct(self, input, future = 0):
        outputs = []
        h_t = mint.zeros(input.size(0), 51, dtype=ms.float64)
        c_t = mint.zeros(input.size(0), 51, dtype=ms.float64)
        h_t2 = mint.zeros(input.size(0), 51, dtype=ms.float64)
        c_t2 = mint.zeros(input.size(0), 51, dtype=ms.float64)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = mint.cat(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    # load data and make training set
    data = torch.load('traindata.pt')  # 'torch.load' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    input = torch.from_numpy(data[3:, :-1])  # 'torch.from_numpy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    target = torch.from_numpy(data[3:, 1:])  # 'torch.from_numpy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    test_input = torch.from_numpy(data[:3, :-1])  # 'torch.from_numpy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    test_target = torch.from_numpy(data[:3, 1:])  # 'torch.from_numpy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)  # 'torch.optim.LBFGS' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    #begin to train
    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        # 'torch.no_grad' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()
