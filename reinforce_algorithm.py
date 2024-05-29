"""
Need to impelemt 
1. Policy Network
2. train one epoch -> log reward and step 
3. train all function
"""
import random 
import math 

import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions.categorical import Categorical
import gym
import wandb

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_state=64):
        super().__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_state),
            nn.Tanh(),
            nn.Linear(hidden_state, action_dim),
        )

    def get_policy(self, state):
        """
        Get the policy probability
        """
        logits = self.network(state)
        return logits.softmax(dim=-1)

    def get_action(self, state):
        """
        Get the action for the given state
        """
        current_policy = self.get_policy(state)
        dist = Categorical(current_policy)
        action = dist.sample()
        return action.item()
    

def loss_function(network, state, action, reward):
    policy_prob = Categorical(network.get_policy(state))
    log_prob = policy_prob.log_prob(action)
    loss = - log_prob * reward
    return loss.mean()

def train_one_epoch(env, network, optimizer, batch_size = 5000,render=False):
    s, _ = env.reset()
    iter = 0
    current_reward = 0
    current_ep_len = 0
    total_reward = []
    total_state = []
    total_action = []
    reward_history = []
    ep_len_history = []
    while True:
        with torch.no_grad():
            a = network.get_action(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
        next_s, r, done, _, _ = env.step(a)
        total_state.append(s)
        total_action.append(a)
        current_ep_len += 1
        current_reward += r 
        s = next_s 
        iter += 1
        if done:
            total_reward += [current_reward] * current_ep_len  
            reward_history.append(current_reward)
            ep_len_history.append(current_ep_len)
            if iter > batch_size:
                break
            s = env.reset()[0]
            current_reward = 0
            current_ep_len = 0
    total_state = torch.as_tensor(total_state, dtype=torch.float32)
    total_action = torch.as_tensor(total_action, dtype=torch.int64)
    total_reward = torch.as_tensor(total_reward, dtype=torch.float32)
    # forward pass
    loss = loss_function(network, total_state, total_action, total_reward)
    # backward pass 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(), np.mean(reward_history), np.mean(ep_len_history)


def train(env, lr=1e-2, num_epochs = 50, batch_size=5000, render=False):
    env = gym.make(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    assert type(action_dim) == int, "Only for discreate action space"
    network = PolicyNetwork(state_dim, action_dim)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    for epoch in range(num_epochs):
        avg_loss, avg_reward, avg_ep_len = train_one_epoch(env, network, optimizer, batch_size, render)
        wandb.log({"loss": avg_loss, "reward": avg_reward})
        print("Epoch: {}, Loss: {}, Average Reward: {}, Average Ep Len: {}".format(epoch, avg_loss, avg_reward, avg_ep_len))
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    wandb.finish()
    wandb.init(project="reinforce", name="REINFORCE")
    print("\n Test REINFORCE algorithm - simplest policy gradient. \n")
    train(args.env_name, lr=args.lr, render=args.render)
    wandb.finish()