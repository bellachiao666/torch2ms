import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import os
import threading
from datetime import datetime
import warnings

# import torch
# import torch.distributed as dist
# import torch.distributed.rpc as rpc
# import torch.multiprocessing as mp
# import torch.nn as nn
# from torch import optim

# import torchvision

# Suppress deprecated ProcessGroup warning
warnings.filterwarnings("ignore", message="You are using a Backend.*ProcessGroup")


batch_size = 20
image_w = 64
image_h = 64
num_classes = 30
batch_update_size = 5
num_batches = 6


def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


class BatchUpdateParameterServer(object):

    def __init__(self, batch_update_size=batch_update_size):
        self.model = torchvision.models.resnet50(num_classes=num_classes)  # 'torchvision.models.resnet50' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()  # 'torch.futures.Future' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)  # 'torch.optim.SGD' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        for p in self.model.parameters():
            p.grad = mint.zeros_like(p)

    def get_model(self):
        return self.model

    # 装饰器 'torch.distributed.rpc.functions.async_execution' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        timed_log(f"PS got {self.curr_update_size}/{batch_update_size} updates")
        for p, g in zip(self.model.parameters(), grads):
            p.grad += g
        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=False)
                fut.set_result(self.model)
                timed_log("PS updated model")
                self.future_model = torch.futures.Future()  # 'torch.futures.Future' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        return fut


class Trainer(object):

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.MSELoss()
        self.one_hot_indices = torch.LongTensor(batch_size) \
                                    .random_(0, num_classes) \
                                    .view(batch_size, 1)  # 'torch.LongTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.LongTensor.random_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.LongTensor.random_.view' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def get_next_batch(self):
        for _ in range(num_batches):
            inputs = mint.randn(batch_size, 3, image_w, image_h)
            labels = mint.zeros(batch_size, num_classes) \
                        .scatter_(1, self.one_hot_indices, 1)
            yield inputs.cuda(), labels.cuda()

    def train(self):
        name = rpc.get_worker_info().name  # 'torch.distributed.rpc.get_worker_info' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        m = self.ps_rref.rpc_sync().get_model().cuda()
        for inputs, labels in self.get_next_batch():
            timed_log(f"{name} processing one batch")
            self.loss_fn(m(inputs), labels).backward()
            timed_log(f"{name} reporting grads")
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            ).cuda()  # 'torch.distributed.rpc.rpc_sync' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.distributed.rpc.rpc_sync.cuda' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            timed_log(f"{name} got updated model")


def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    trainer.train()


def run_ps(trainers):
    timed_log("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer())  # 'torch.distributed.rpc.RRef' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )  # 'torch.distributed.rpc.rpc_async' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    torch.futures.wait_all(futs)  # 'torch.futures.wait_all' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    timed_log("Finish training")


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group first
    mint.distributed.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )
    
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=60
     )  # 'torch.distributed.rpc.TensorPipeRpcBackendOptions' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        run_ps([f"trainer{r}" for r in range(1, world_size)])

    # block until all rpcs finish
    rpc.shutdown()  # 'torch.distributed.rpc.shutdown' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    dist.destroy_process_group()  # 'torch.distributed.destroy_process_group' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


if __name__=="__main__":
    world_size = batch_update_size + 1
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)  # 'torch.multiprocessing.spawn' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
