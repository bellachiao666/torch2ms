import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import logging
# import torch

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

def get_logger():
    return logging.getLogger(__name__)


def rank_log(_rank, logger, msg):
    """helper function to log only on global rank 0"""
    if _rank == 0:
        logger.info(f" {msg}")


def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()  # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    gpu_count = torch.accelerator.device_count()  # 'torch.accelerator.device_count' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    return has_gpu and gpu_count >= min_gpus
