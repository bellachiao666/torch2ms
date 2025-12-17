import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import argparse
import gymnasium as gym
import numpy as np
import os
from itertools import count

# import torch
# import torch.distributed.rpc as rpc
# import torch.multiprocessing as mp
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
# from torch.distributions import Categorical

TOTAL_EPISODE_STEP = 5000
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

parser = argparse.ArgumentParser(description='PyTorch RPC RL example')
parser.add_argument('--world-size', type=int, default=2, metavar='W',
                    help='world size for RPC, rank 0 is the agent, others are observers')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)  # 'torch.distributed.rpc.rpc_sync' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


class Policy(msnn.Cell):
    r"""
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/main/reinforcement_learning
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p = 0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def construct(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        action_scores = self.affine2(x)
        return nn.functional.softmax(action_scores, dim = 1)

class Observer:
    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.

    It is true that CartPole-v1 is a relatively inexpensive environment, and it
    might be an overkill to use RPC to connect observers and trainers in this
    specific use case. However, the main goal of this tutorial to how to build
    an application using the RPC API. Developers can extend the similar idea to
    other applications with much more expensive environment.
    """
    def __init__(self):
        self.id = rpc.get_worker_info().id  # 'torch.distributed.rpc.get_worker_info' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.env = gym.make('CartPole-v1')
        self.env.reset(seed=args.seed)

    def run_episode(self, agent_rref, n_steps):
        r"""
        Run one episode of n_steps.

        Args:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        state, ep_reward = self.env.reset()[0], 0
        for step in range(n_steps):
            # send the state to the agent to get an action
            action = _remote_method(Agent.select_action, agent_rref, self.id, state)

            # apply the action to the environment, and get the reward
            state, reward, terminated, truncated, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)

            if terminated or truncated:
                break

class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)  # 'torch.distributed.rpc.RRef' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = mint.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))  # 'torch.distributed.rpc.get_worker_info' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            self.ob_rrefs.append(remote(ob_info, Observer))  # 'torch.distributed.rpc.remote' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, ob_id, state):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/main/reinforcement_learning
        The main difference is that instead of keeping all probs in one list,
        the agent keeps probs in a dictionary, one key per observer.

        NB: no need to enforce thread-safety here as GIL will serialize
        executions.
        """
        state = ms.float32().unsqueeze(0)  # 'torch.from_numpy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        probs = self.policy(state)
        m = Categorical(probs)  # 'torch.distributions.Categorical' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        r"""
        Observers call this function to report rewards.
        """
        self.rewards[ob_id].append(reward)

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each oberser to run n_steps.
        """
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps)
                )
            )  # 'torch.distributed.rpc.rpc_async' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

    def finish_episode(self):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/main/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """

        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

        # clear saved probs and rewards
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = ms.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = mint.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        return min_reward


def run_worker(rank, world_size):
    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        agent = Agent(world_size)
        for i_episode in count(1):
            n_steps = int(TOTAL_EPISODE_STEP / (args.world_size - 1))
            agent.run_episode(n_steps=n_steps)
            last_reward = agent.finish_episode()

            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, last_reward, agent.running_reward))

            if agent.running_reward > agent.reward_threshold:
                print("Solved! Running reward is now {}!".format(agent.running_reward))
                break
    else:
        # other ranks are the observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)  # 'torch.distributed.rpc.init_rpc' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # observers passively waiting for instructions from agents
    rpc.shutdown()  # 'torch.distributed.rpc.shutdown' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def main():
    mp.spawn(
        run_worker,
        args=(args.world_size, ),
        nprocs=args.world_size,
        join=True
    )  # 'torch.multiprocessing.spawn' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

if __name__ == '__main__':
    main()
