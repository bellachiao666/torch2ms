import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import argparse
import os

# import torch
from checkpoint import Checkpointer
from model import ModelArgs, Transformer
# from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from utils import inspect_mixed_precision, inspect_model

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()  # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    gpu_count = torch.accelerator.device_count()  # 'torch.accelerator.device_count' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    return has_gpu and gpu_count >= min_gpus

def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        exit()
    rank = int(os.environ["LOCAL_RANK"])
    # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        device = torch.device(f"{device_type}:{rank}")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch.accelerator.device_index(rank)  # 'torch.accelerator.device_index' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        print(f"Running on rank {rank} on device {device}")
    else:
        device = torch.device("cpu")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        print(f"Running on device {device}")

    backend = torch.distributed.get_default_backend_for_device(device)  # 'torch.distributed.get_default_backend_for_device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    mint.distributed.init_process_group(backend=backend, device_id=device)

    torch.manual_seed(0)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=10,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    with torch.device("meta"):
        model = Transformer(model_args)
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=ms.bfloat16,
            reduce_dtype=ms.float32,
        )  # 'torch.distributed.fsdp.MixedPrecisionPolicy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)  # 'torch.distributed.fsdp.fully_shard' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 存在 *args/**kwargs，未转换，需手动确认参数映射;
    fully_shard(model, **fsdp_kwargs)  # 'torch.distributed.fsdp.fully_shard' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 存在 *args/**kwargs，未转换，需手动确认参数映射;

    inspect_model(model)

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is None:
        model.to_empty(device=device)
        model.reset_parameters()
    else:
        checkpointer.load_model(model)
    
    if args.mixed_precision:
        inspect_mixed_precision(model)

    optim = mint.optim.Adam(model.parameters(), lr=1e-2)
    if checkpointer.last_training_time is not None:
        checkpointer.load_optim(model, optim)

    for _ in range(10):
        if args.explicit_prefetching:
            model.unshard()
        x = mint.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 'torch.nn.utils.clip_grad_norm_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        optim.step()
        optim.zero_grad()

    checkpointer.save(model, optim)
    torch.distributed.destroy_process_group()  # 'torch.distributed.destroy_process_group' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    args = parser.parse_args()
    
    main(args)
