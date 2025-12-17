import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)  # 'torch.cuda.set_device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    mint.distributed.init_process_group(backend = "nccl", world_size = world_size, rank = rank)

class Trainer:
    # 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    # 'torch.optim.Optimizer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    def __init__(
        self,
        model: msnn.Cell,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])  # 'torch.nn.parallel.DistributedDataParallel' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)  # 'torch.nn.functional.cross_entropy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)  # 'torch.save' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 'torch.optim.SGD' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    return train_set, model, optimizer


# 'torch.utils.data.Dataset' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )  # 'torch.utils.data.distributed.DistributedSampler' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()  # 'torch.distributed.destroy_process_group' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()  # 'torch.cuda.device_count' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)  # 'torch.multiprocessing.spawn' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
