"""
Base Code from : https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
Policy gradient on cartpole
"""

import argparse
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v1')
env = gym.make('MountainCar-v0')
env = gym.make('Acrobot-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(env.observation_space.high.__len__(), 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, env.action_space.n)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(state):
    """
    Returns action for a given state.
    The random action selection is weighted according to Q-values of actions.
    """
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()  # Sample the action space based on probability computed by softmax
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    """
    Compute loss and back propagate
    """
    R = 0
    policy_loss = []
    returns = []

    # The rewards obtained later in the episode are valued more
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    # Normalizing retuns with mean 0 and sigma 1
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # Compute negative log likelihood loss
    for log_prob, R in zip(policy.saved_log_probs, returns):
        # log_prob of sampled action * reward
        policy_loss.append(-log_prob * R)
    # Convert policy_loss from list to tensor and compute summation
    policy_loss = torch.cat(policy_loss).sum()

    # Backpropagate and update weights of the policy gradient network
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # Empty rewards and saved log prob for next episode
    policy.rewards = []
    policy.saved_log_probs = []


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        while True:
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {:.2f}".format(running_reward))
        #     break

        if i_episode > 100:
            env.render()

    env.close()


if __name__ == '__main__':
    main()
