import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import argparse
import os
from threading import Lock

# import torch
# import torch.distributed.autograd as dist_autograd
# import torch.distributed.rpc as rpc
# import torch.multiprocessing as mp
# import torch.nn as nn
# from torch import optim
# from torch.distributed.optim import DistributedOptimizer
# from torchvision import datasets, transforms

# --------- MNIST Network to train, from pytorch/examples -----


class Net(msnn.Cell):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        print(f"Using {num_gpus} GPUs to train")
        self.num_gpus = num_gpus
        # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if torch.accelerator.is_available() and self.num_gpus > 0:
            acc = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            device = torch.device(f'{acc}:0')  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            device = torch.device("cpu")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        print(f"Putting first 2 convs on {str(device)}")
        # Put conv layers on the first accelerator device
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)  # 'torch.nn.Conv2d.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)  # 'torch.nn.Conv2d.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # Put rest of the network on the 2nd accelerator device, if there is one
        # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if torch.accelerator.is_available() and self.num_gpus > 0:
            acc = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            device = torch.device(f'{acc}:1')  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        print(f"Putting rest of layers on {str(device)}")
        self.dropout1 = nn.Dropout2d(0.25).to(device)  # 'torch.nn.Dropout2d.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.dropout2 = nn.Dropout2d(0.5).to(device)  # 'torch.nn.Dropout2d.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.fc1 = nn.Linear(9216, 128).to(device)  # 'torch.nn.Linear.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.fc2 = nn.Linear(128, 10).to(device)  # 'torch.nn.Linear.to' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = mint.flatten(x, 1)
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = mint.special.log_softmax(x, dim=1)
        return output


# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)  # 'torch.distributed.rpc.rpc_sync' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

# --------- Parameter Server --------------------
class ParameterServer(msnn.Cell):
    def __init__(self, num_gpus=0):
        super().__init__()
        model = Net(num_gpus=num_gpus)
        self.model = model
        # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if torch.accelerator.is_available() and num_gpus > 0:
            acc = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            self.input_device = torch.device(f'{acc}:0')  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            self.input_device = torch.device("cpu")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            
    def construct(self, inp):
        inp = inp.to(self.input_device)
        out = self.model(inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to("cpu")
        return out

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)  # 'torch.distributed.autograd.get_gradients' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes parameters remotely.
    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]  # 'torch.distributed.rpc.RRef' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        return param_rrefs

param_server = None
global_lock = Lock()

def get_parameter_server(num_gpus=0):
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()  # 'torch.distributed.rpc.shutdown' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    print("RPC shutdown on parameter server.")


# --------- Trainers --------------------

# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(msnn.Cell):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(num_gpus,))  # 'torch.distributed.rpc.remote' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params

    def construct(self, x):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output


def run_training_loop(rank, num_gpus, train_loader, test_loader):
    # Runs the typical neural network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet(num_gpus=num_gpus)
    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)  # 'torch.distributed.optim.DistributedOptimizer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    for i, (data, target) in enumerate(train_loader):
        # 'torch.distributed.autograd.context' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        with dist_autograd.context() as cid:
            model_output = net(data)
            target = target.to(model_output.device)
            loss = nn.functional.nll_loss(model_output, target)
            if i % 5 == 0:
                print(f"Rank {rank} training batch {i} loss {loss.item()}")
            dist_autograd.backward(cid, [loss])  # 'torch.distributed.autograd.backward' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            # Ensure that dist autograd ran successfully and gradients were
            # returned.
            assert remote_method(
                ParameterServer.get_dist_gradients,
                net.param_server_rref,
                cid) != {}
            opt.step(cid)

    print("Training complete!")
    print("Getting accuracy....")
    get_accuracy(test_loader, net)


def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    if torch.accelerator.is_available() and model.num_gpus > 0:
        acc = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        device = torch.device(f'{acc}:0')  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    else:
        device = torch.device("cpu")  # 'torch.device' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    # 'torch.no_grad' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")


# Main loop for trainers.
def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    print(f"Worker {rank} done initializing RPC")

    run_training_loop(rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()  # 'torch.distributed.rpc.shutdown' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

# --------- Launcher --------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    processes = []
    world_size = args.world_size

    # Note that Linux uses "fork" by default, which may cause deadlock.
    # Besides, cuda doesn't support "fork" and Windows only supports "spawn"
    mp.set_start_method("spawn")  # 'torch.multiprocessing.set_start_method' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    if args.rank == 0:
        p = mp.Process(target=run_parameter_server, args=(0, world_size))  # 'torch.multiprocessing.Process' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        p.start()
        processes.append(p)
    else:
        # Get data to train on
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=ms.dataset.transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=32, shuffle=True)  # 'torchvision.transforms.ToTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.transforms.Normalize' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.datasets.MNIST' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                           transform=ms.dataset.transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=32, shuffle=True)  # 'torchvision.transforms.ToTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.transforms.Normalize' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torchvision.datasets.MNIST' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.utils.data.DataLoader' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # start training worker on this node
        p = mp.Process(
            target=run_worker,
            args=(
                args.rank,
                world_size, args.num_gpus,
                train_loader,
                test_loader))  # 'torch.multiprocessing.Process' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
