import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import os

# import torch
# import torch.distributed.autograd as dist_autograd
# import torch.distributed.rpc as rpc
# import torch.multiprocessing as mp
# import torch.optim as optim
# from torch.distributed.optim import DistributedOptimizer

import rnn


def _run_trainer():
    r"""
    The trainer creates a distributed RNNModel and a DistributedOptimizer. Then,
    it performs training using random input data.
    """
    batch = 5
    ntoken = 7
    ninp = 2

    nhid = 3
    nindices = 6
    nlayers = 4
    hidden = (
        mint.randn(nlayers, nindices, nhid),
        mint.randn(nlayers, nindices, nhid)
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )  # 'torch.distributed.optim.DistributedOptimizer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    criterion = nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken  # 'torch.LongTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            target = torch.LongTensor(batch, ntoken) % nindices  # 'torch.LongTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            yield data, target

    # train for 10 iterations
    for epoch in range(10):
        # create distributed autograd context
        for data, target in get_next_batch():
            # 'torch.distributed.autograd.context' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            with dist_autograd.context() as context_id:
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss])  # 'torch.distributed.autograd.backward' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
                # run distributed optimizer
                opt.step(context_id)
                # not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training epoch {}".format(epoch))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        _run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # parameter server does nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()  # 'torch.distributed.rpc.shutdown' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)  # 'torch.multiprocessing.spawn' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
