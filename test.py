import argparse
import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from torch.optim import RMSprop


class RainEnv:  # 簡單的雨量預測環境
    def __init__(self, csv_path, window_size=5):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.fillna(method='ffill')
        self.window_size = window_size
        self.cur = 0
        self.max_index = len(self.data) - 1

    def reset(self):
        self.cur = random.randint(0, self.max_index - self.window_size - 1)
        return self._get_obs()

    def step(self, action=None):
        self.cur += 1
        done = self.cur + self.window_size >= self.max_index
        obs = self._get_obs()
        reward = self.data.iloc[self.cur + self.window_size]["Precipitation"]
        return obs, reward, done, {}

    def _get_obs(self):
        obs = self.data.iloc[self.cur:self.cur + self.window_size]
        return obs.drop(columns=["Precipitation"]).values.flatten()


class VPNNetwork(nn.Module):
    def __init__(self, input_dim, option_dim):
        super(VPNNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + option_dim, 64),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(64, 1)
        self.discount_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.transition_head = nn.Linear(64, input_dim)

    def forward(self, state, option):
        x = torch.cat([state, option], dim=-1)
        x = self.encoder(x)
        reward = self.reward_head(x)
        discount = torch.sigmoid(self.discount_head(x))
        value = self.value_head(x)
        next_state = self.transition_head(x)
        return reward, discount, value, next_state


def get_best_options(net, state, b=10):
    return [torch.eye(3)[i].unsqueeze(0) for i in range(3)]


class VPN:
    def __init__(self, network_fn, env_fn, d, k, n, max_memory, seed=0):
        self.network_fn = network_fn
        self.env_fn = env_fn
        self.global_network = network_fn()
        self.target_network = network_fn()
        self.global_t = 0
        self.t = 0
        self.d = d
        self.k = k
        self.n = n
        self.max_memory = max_memory
        self.memory = []
        self.global_optimizer = RMSprop(self.global_network.parameters(), lr=1e-3)
        self.seed = seed
        self._stop_training = False

    def init_memory(self):
        self.memory = []

    def update_global_grads(self, net):
        for param, shared_param in zip(net.parameters(), self.global_network.parameters()):
            if shared_param.grad is not None:
                continue
            shared_param._grad = param.grad

    def _train(self, rank):
        torch.manual_seed(self.seed + rank)
        net = self.network_fn()
        net.load_state_dict(self.global_network.state_dict())
        env = self.env_fn()
        obs = env.reset()
        done = False
        rewards, discounts = [], []
        t_start = self.t

        while not done or (self.t - t_start) < self.n:
            option = torch.eye(3)[random.randint(0, 2)].unsqueeze(0)
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            reward, discount, value, next_state = net(state, option)
            reward = reward.item()
            discount = discount.item()
            rewards.append(reward)
            discounts.append(discount)
            obs, true_reward, done, _ = env.step()
            self.t += 1
            self.global_t += 1

        R = 0 if done else max(rewards)
        loss = 0
        for i in range(len(rewards) - 1, -1, -1):
            R = rewards[i] + discounts[i] * R
            value_loss = (R - value) ** 2
            loss += value_loss

        self.global_optimizer.zero_grad()
        loss.backward()
        self.update_global_grads(net)
        self.global_optimizer.step()

        if rank == 0 and self.global_t % 10 == 0:
            self.target_network.load_state_dict(self.global_network.state_dict())

    def train(self, num_processes=2):
        self.init_memory()
        self.global_t = 0
        self.t = 0
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=self._train, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print("Training complete.")

    def test(self):
        self._stop_training = True

    def predict(self, input_obs):
        with torch.no_grad():
            option = torch.eye(3)[0].unsqueeze(0)
            state = torch.tensor(input_obs, dtype=torch.float32).unsqueeze(0)
            reward, discount, value, next_state = self.global_network(state, option)
            return reward.item(), discount.item(), value.item()


def env_fn():
    return RainEnv("Masters/Master_Hsinchu.csv", window_size=5)


def network_fn():
    input_dim = (12 - 1) * 5  # 12欄減掉 Precipitation，乘上 window_size
    option_dim = 3
    return VPNNetwork(input_dim, option_dim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VPN Rainfall Prediction')
    parser.add_argument('--train', action='store_true', help='Train the VPN model')
    parser.add_argument('--predict', action='store_true', help='Predict rainfall')
    args = parser.parse_args()

    vpn = VPN(network_fn, env_fn, d=2, k=3, n=5, max_memory=1000)

    if args.train:
        vpn.train(num_processes=2)

    if args.predict:
        env = env_fn()
        obs = env.reset()
        reward, discount, value = vpn.predict(obs)
        print(f"[Prediction] Reward: {reward:.2f}, Discount: {discount:.2f}, Value: {value:.2f}")
