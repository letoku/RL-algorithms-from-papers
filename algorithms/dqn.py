"""
self contained example - both implementation and simple test. Not coherent to other algorithms in repository,\
because I wrote this DQN before this project. But I found it somewhere in my files, so I added it to this project.
"""



import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import copy
from sklearn.preprocessing import MinMaxScaler


import torch
import torch.nn as nn


if gym.__version__ < '0.26':
    env = gym.make('CartPole-v0', new_step_api=True, render_mode='single_rgb_array').unwrapped
else:
    env = gym.make('CartPole-v0', render_mode='rgb_array').unwrapped


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminated'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_units: int):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, 1)

    def forward(self, layer_1_input):
        layer_2_input = self.lin1(layer_1_input).relu()
        y = self.lin2(layer_2_input)

        return y


def get_state(obs):
    state = scaler.transform(obs.reshape(1, -1))[0]

    return state


def choose_action(q_network, state, epsilon):
    if np.random.uniform() < epsilon:
        action = np.random.choice(range(2))
    else:
        q_network.eval()
        torch.set_grad_enabled(False)

        q_0_option = q_network(torch.tensor(np.append(state, 0), dtype=torch.float).to(device)).item()
        q_1_option = q_network(torch.tensor(np.append(state, 1), dtype=torch.float).to(device)).item()

        if q_0_option >= q_1_option:
            action = 0
        else:
            action = 1

    return action


def benchmark():
    terminated = False
    episode_score = 0
    eps = 0
    obs = env.reset()[0]

    while not terminated:
        state = get_state(obs)
        action = choose_action(q_network, state, eps)

        obs, reward, terminated, info, _ = env.step(action)

        episode_score += reward

    return episode_score


def perform_gradient_descent(q_network, q_target, dataset, batch_size):
    transitions_batch = dataset.sample(batch_size=batch_size)
    X = np.array([np.append(transition.state, transition.action) for transition in transitions_batch])
    X = torch.tensor(X, dtype=torch.float).to(device)

    # for q-target we don't need to calculate gradients
    q_target.eval()
    torch.set_grad_enabled(False)

    y = torch.tensor(
        [
            max(
                transition.reward + gamma * q_target(
                    torch.tensor(np.append(transition.next_state, 0), dtype=torch.float).to(device)).item(),
                transition.reward + gamma * q_target(
                    torch.tensor(np.append(transition.next_state, 1), dtype=torch.float).to(device)).item()
            )
            if not transition.terminated else transition.reward

            for transition in transitions_batch
        ],
        dtype=torch.float).to(device)

    q_network.train()
    torch.set_grad_enabled(True)
    pred = q_network(X)
    loss = loss_fn(pred, y.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return q_network


cart_pos_lower_bound = -4.8
cart_pos_upper_bound = 4.8

cart_v_lower_bound = -4.8
cart_v_upper_bound = 4.8

pole_angle_lower_bound = math.radians(-24)
pole_angle_upper_bound = math.radians(24)

pole_omega_lower_bound = math.radians(-50)
pole_omega_upper_bound = math.radians(50)

matrix_to_fit = np.asarray([
    [cart_pos_lower_bound, cart_pos_upper_bound],
    [cart_v_lower_bound, cart_v_upper_bound],
    [pole_angle_lower_bound, pole_angle_upper_bound],
    [pole_omega_lower_bound, pole_omega_upper_bound]
]).T

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(matrix_to_fit)

# matrix_to_transform = np.asarray([
#     [-1220],
#     [3],
#     [0],
#     [-3]
# ]).T

# scaler.transform(matrix_to_transform)


torch.manual_seed(0)
np.random.seed(0)


num_of_episodes = 35000
epsilon = 0.1
gamma = 0.9
alpha = 0.2  # learning rate
c = 1000  # number of steps after we copy q network to target network
dataset_size = 10000
learning_rate = 0.001  # for gradient descent on q-network
batch_size = 10

q_network = DQN(input_dim=5, hidden_units=10).to(device)
q_target = copy.deepcopy(DQN(input_dim=5, hidden_units=10).to(device))

optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

dataset = ReplayMemory(capacity=dataset_size)

scores = []
counter = 0
for episode in range(num_of_episodes):
    obs = env.reset()[0]
    terminated = False
    episode_score = 0
    eps = 0.5 * (1 - episode / num_of_episodes)

    while not terminated:
        state = get_state(obs)
        action = choose_action(q_network, state, eps)

        obs, reward, terminated, info, _ = env.step(action)
        new_state = get_state(obs)

        dataset.push(Transition(state=state, action=action, next_state=new_state, reward=reward, terminated=terminated))

        old_q_network = copy.deepcopy(q_network)
        q_network = perform_gradient_descent(q_network=q_network, q_target=q_target, dataset=dataset, batch_size=batch_size)

        episode_score += reward
        counter += 1

        if counter == c:
            # rewriting current network to target one
            q_target = copy.deepcopy(q_network)
            counter = 0

    scores.append(benchmark())
    if (episode + 1) % 50 == 0:
        print('---------------')
        print(f'Episode {episode + 1} score: {benchmark()}')
