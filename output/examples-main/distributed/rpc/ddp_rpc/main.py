import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import random

# import torch
# import torch.distributed as dist
# import torch.distributed.autograd as dist_autograd
# import torch.distributed.rpc as rpc
# import torch.multiprocessing as mp
# import torch.optim as optim
# from torch.distributed.nn import RemoteModule
# from torch.distributed.optim import DistributedOptimizer
# from torch.distributed.rpc import RRef
# from torch.distributed.rpc import TensorPipeRpcBackendOptions
# from torch.nn.parallel import DistributedDataParallel as DDP

NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()  # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    gpu_count = torch.accelerator.device_count()  # 'torch.accelerator.device_count' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    return has_gpu and gpu_count >= min_gpus

class HybridModel(msnn.Cell):
    r"""
    The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """

    def __init__(self, remote_emb_module, rank):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(nn.Linear(16, 8).to(rank))  # 'torch.nn.parallel.DistributedDataParallel' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.rank = rank

    def construct(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup.to(self.rank))


def _run_trainer(remote_emb_module, rank):
    r"""
    Each trainer runs a forward pass which involves an embedding lookup on the
    parameter server and running nn.Linear locally. During the backward pass,
    DDP is responsible for aggregating the gradients for the dense part
    (nn.Linear) and distributed autograd ensures gradients updates are
    propagated to the parameter server.
    """

    # Setup the model.
    model = HybridModel(remote_emb_module, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.

    # Retrieve parameters for embedding table.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    # model.fc.parameters() only includes local parameters.
    # NOTE: Cannot call model.parameters() here,
    # because this will call remote_emb_module.parameters(),
    # which supports remote_parameters() but not parameters().
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))  # 'torch.distributed.rpc.RRef' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    # Setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )  # 'torch.distributed.optim.DistributedOptimizer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    criterion = nn.CrossEntropyLoss()

    def get_next_batch(rank):
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)  # 'torch.LongTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.LongTensor.random_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

            # Generate offsets.
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)  # 'torch.LongTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            target = torch.LongTensor(batch_size).random_(8).to(rank)  # 'torch.LongTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.LongTensor.random_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.LongTensor.random_.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            yield indices, offsets_tensor, target

    # Train for 100 epochs
    for epoch in range(100):
        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):
            # 'torch.distributed.autograd.context' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])  # 'torch.distributed.autograd.backward' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

                # Tun distributed optimizer
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training done for epoch {}".format(epoch))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """

    # We need to use different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()  # 'torch.distributed.rpc.TensorPipeRpcBackendOptions' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
    if rank == 2:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )  # 'torch.distributed.nn.RemoteModule' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        # Run the training loop on trainers.
        futs = []
        for trainer_rank in [0, 1]:
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank)
            )  # 'torch.distributed.rpc.rpc_async' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            futs.append(fut)

        # Wait for all training to finish.
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        acc = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        device = torch.device(acc)  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        backend = torch.distributed.get_default_backend_for_device(device)  # 'torch.distributed.get_default_backend_for_device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch.accelerator.device_index(rank)  # 'torch.accelerator.device_index' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # Initialize process group for Distributed DataParallel on trainers.
        mint.distributed.init_process_group(
            backend=backend, rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # Initialize RPC.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        # Trainer just waits for RPCs from master.
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()  # 'torch.distributed.rpc.shutdown' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    
    # Clean up process group for trainers to avoid resource leaks
    if rank <= 1:
        dist.destroy_process_group()  # 'torch.distributed.destroy_process_group' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


if __name__ == "__main__":
    # 2 trainers, 1 parameter server, 1 master.
    world_size = 4
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        exit()
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)  # 'torch.multiprocessing.spawn' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    print("Distributed RPC example completed successfully.")
