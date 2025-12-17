import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


render_mode = "human" if args.render else None
env = gym.make('CartPole-v1', render_mode=render_mode)
env.reset(seed=args.seed)
torch.manual_seed(args.seed)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


class Policy(msnn.Cell):
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


policy = Policy()
optimizer = mint.optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = ms.float32().unsqueeze(0)  # 'torch.from_numpy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    probs = policy(state)
    m = Categorical(probs)  # 'torch.distributions.Categorical' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = ms.Tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = mint.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if terminated or truncated:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward} and the last episode runs to {t} time steps!")
            break


if __name__ == '__main__':
    main()
