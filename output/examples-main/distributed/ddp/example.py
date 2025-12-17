import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse

# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim

# from torch.nn.parallel import DistributedDataParallel as DDP

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()  # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    gpu_count = torch.accelerator.device_count()  # 'torch.accelerator.device_count' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    return has_gpu and gpu_count >= min_gpus

class ToyModel(msnn.Cell):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def construct(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank):

    print(
        f"[{os.getpid()}] rank = {mint.distributed.get_rank()}, "
        + f"world_size = {mint.distributed.get_world_size()}"
        )

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])  # 'torch.nn.parallel.DistributedDataParallel' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)  # 'torch.optim.SGD' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    optimizer.zero_grad()
    outputs = ddp_model(mint.randn(20, 10))
    labels = mint.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    print(f"training completed in rank {rank}!")


def main():
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    rank = int(env_dict['RANK'])
    local_rank = int(env_dict['LOCAL_RANK'])
    local_world_size = int(env_dict['LOCAL_WORLD_SIZE'])
    
    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        if "INIT_METHOD" in os.environ.keys():
            print(f"init_method is {os.environ['INIT_METHOD']}")
            url_obj = urlparse(os.environ["INIT_METHOD"])
            if url_obj.scheme.lower() != "file":
                raise ValueError("Windows only supports FileStore")
            else:
                init_method = os.environ["INIT_METHOD"]
        else:
            # It is a example application, For convience, we create a file in temp dir.
            temp_dir = tempfile.gettempdir()
            init_method = f"file:///{os.path.join(temp_dir, 'ddp_example')}"
        mint.distributed.init_process_group(backend="gloo", init_method=init_method, rank=int(env_dict["RANK"]), world_size=int(env_dict["WORLD_SIZE"]))
    else:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
        acc = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        backend = torch.distributed.get_default_backend_for_device(acc)  # 'torch.distributed.get_default_backend_for_device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch.accelerator.set_device_index(rank)  # 'torch.accelerator.set_device_index' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        mint.distributed.init_process_group(backend=backend)

    print(
        f"[{os.getpid()}]: world_size = {mint.distributed.get_world_size()}, "
        + f"rank = {mint.distributed.get_rank()}, backend={mint.distributed.get_backend()} \n", end=''
    )

    demo_basic(rank)

    # Tear down the process group
    dist.destroy_process_group()  # 'torch.distributed.destroy_process_group' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

if __name__ == "__main__":
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        sys.exit()
    main()
