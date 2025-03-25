import numpy as np
import copy
import torch
import torch.nn as nn

from abstract_classes import Agent
from structures.replay_memory import ReplayMemory


class Actor(torch.nn.Module):
    def __init__(self,
                 states_dim: int,
                 hidden_units: int,
                 action_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(states_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        layer_2_input = self.lin1(state).relu()
        y = self.lin2(layer_2_input)

        return y


class Critic(torch.nn.Module):
    def __init__(self,
                 states_dim: int,
                 action_dim: int,
                 hidden_units: int,):
        super().__init__()
        self.lin1 = nn.Linear(states_dim + action_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat(tensors=(state, action), dim=1)

        layer_2_input = self.lin1(critic_input).relu()
        y = self.lin2(layer_2_input)

        return y


class DDPG(Agent):
    def __init__(self, memory_size: int, states_dim: int, actions_dim: int, action_min: float,
                 action_max: float, noise_sigma: float, polyak: float, gamma: float, network_hidden_units: int,
                 batch_size: int, q_learning_rate: float, pi_learning_rate: float, device):

        """actions parameters"""
        self.noise_sigma = noise_sigma
        self.action_dim = actions_dim
        self.action_min = action_min
        self.action_max = action_max

        """training hyperparameters"""
        self.polyak = polyak
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_learning_rate = q_learning_rate
        self.pi_learning_rate = pi_learning_rate

        self.loss_fn = torch.nn.MSELoss()

        """agent's approximators"""
        # q approximators
        self.Q = Critic(states_dim=states_dim, action_dim=actions_dim, hidden_units=network_hidden_units).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.q_learning_rate)

        self.target_Q = copy.deepcopy(self.Q)
        self.target_Q.eval()

        # policy approximators
        self.policy = Actor(states_dim=states_dim, action_dim=actions_dim, hidden_units=network_hidden_units).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.pi_learning_rate)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_policy.eval()

        for p in self.target_Q.parameters():
            p.requires_grad = False
        for p in self.target_policy.parameters():
            p.requires_grad = False

        self.device = device
        self.memory = ReplayMemory(capacity=memory_size, states_dim=states_dim, actions_dim=actions_dim, device=device)

    def get_action(self, state: np.ndarray, noise: float = 0.1, exploratory_phase: bool = False) -> np.ndarray:
        if exploratory_phase:
            return np.random.uniform(self.action_min, self.action_max, size=(1, self.action_dim))

        policy_input = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.policy(policy_input)

        action = action.cpu().detach().numpy()
        action = np.add(action, np.random.normal(0, noise, size=(1, self.action_dim)))

        return np.clip(action, a_min=self.action_min, a_max=self.action_max)

    def _create_targets(self, batch):
        target_actions = self.target_policy(batch['next_states'])
        estimated_Q_values = self.target_Q(state=batch['next_states'], action=target_actions) * (1-batch['terminated'])

        return torch.add(input=batch['rewards'], other=estimated_Q_values, alpha=self.gamma)

    def _update_Q(self, batch):
        targets = self._create_targets(batch)

        with torch.enable_grad():
            y = self.Q(state=batch['states'], action=batch['actions'])
            loss = self.loss_fn(y, targets)

            self.Q_optimizer.zero_grad()
            loss.backward()
            self.Q_optimizer.step()

    def _update_policy(self, batch):
        with torch.enable_grad():
            y = -self.Q(state=batch['states'], action=self.policy(batch['states'])).mean()

            self.policy_optimizer.zero_grad()
            y.backward()
            self.policy_optimizer.step()

    def update(self):
        batch = self.memory.sample(batch_size=self.batch_size)
        self._update_Q(batch)
        self._update_policy(batch)

        for param, target_param in zip(self.Q.parameters(), self.target_Q.parameters()):
            target_param.data.copy_(self.polyak * param.data + (1 - self.polyak) * target_param.data)

        for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
            target_param.data.copy_(self.polyak * param.data + (1 - self.polyak) * target_param.data)

    def push(self, transition):
        self.memory.push(transition)
