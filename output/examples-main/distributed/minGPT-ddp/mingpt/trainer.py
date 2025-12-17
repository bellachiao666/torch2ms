import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Optional, Any, Dict
import os

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

import boto3
from urllib.parse import urlparse
import fsspec
import io

@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int

def upload_to_s3(obj, dst):
    buffer = io.BytesIO()
    torch.save(obj, buffer)  # 'torch.save' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    buffer.seek(0)
    dst = urlparse(dst, allow_fragments=False)
    boto3.client('s3').upload_fileobj(buffer, dst.netloc, dst.path.lstrip('/'))

class Trainer:

    # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    def __init__(self, trainer_config: TrainerConfig, model, optimizer, train_dataset, test_dataset=None):
        self.config = trainer_config
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])  
        # set device
        self.acc = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.device: torch.device = torch.device(f"{self.acc}:{self.local_rank}")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.device_type = self.device.type
        # data stuff
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None
        # initialize train states
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer        
        self.save_every = self.config.save_every
        if self.config.use_amp:
            self.scaler = torch.amp.GradScaler(self.device_type)  # 'torch.amp.GradScaler' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()
        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])  # 'torch.nn.parallel.DistributedDataParallel' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        
    # 'torch.utils.data.Dataset' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset)
        )  # 'torch.utils.data.distributed.DistributedSampler' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")  # 'torch.load' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return 

        snapshot = Snapshot(**snapshot_data)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _run_batch(self, source, targets, train: bool = True) -> float:
        # 'torch.set_grad_enabled' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # 'torch.amp.autocast' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=self.device_type, dtype=ms.float16, enabled=(self.config.use_amp)):
            _, loss = self.model(source, targets)
        
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_amp: 
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)  # 'torch.nn.utils.clip_grad_norm_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)  # 'torch.nn.utils.clip_grad_norm_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
                self.optimizer.step()
        
        return loss.item()

    # 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        if train:
            dataloader.sampler.set_epoch(epoch)
        for iter, (source, targets) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets, train)
            if iter % 100 == 0:
                print(f"[RANK{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss:.5f}")

    def _save_snapshot(self, epoch):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch
        )
        # save snapshot
        snapshot = asdict(snapshot)
        if self.config.snapshot_path.startswith("s3://"):
            upload_to_s3(snapshot, self.config.snapshot_path)
        else:
            torch.save(snapshot, self.config.snapshot_path)  # 'torch.save' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            
        print(f"Snapshot saved at epoch {epoch}")

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1
            self._run_epoch(epoch, self.train_loader, train=True)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
