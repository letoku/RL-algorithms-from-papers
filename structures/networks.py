import torch
import torch.nn as nn



class Actor(torch.nn.Module):
    def __init__(self,
                 states_dim: int,
                 hidden_units: int,
                 action_dim: int,
                 max_action: float = 1.0):
        super().__init__()
        self.l1 = nn.Linear(states_dim, hidden_units)
        self.l2 = nn.Linear(hidden_units, hidden_units)
        self.l3 = nn.Linear(hidden_units, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        l2_input = self.l1(state).relu()
        l3_input = self.l2(l2_input).relu()
        y = torch.tanh(self.l3(l3_input))*self.max_action

        return y


class Critic(torch.nn.Module):
    def __init__(self,
                 states_dim: int,
                 action_dim: int,
                 hidden_units: int):
        super().__init__()

        self.l1 = nn.Linear(states_dim + action_dim, hidden_units)
        self.l2 = nn.Linear(hidden_units, hidden_units)
        self.l3 = nn.Linear(hidden_units, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat(tensors=(state, action), dim=1)

        l2_input = self.l1(critic_input).relu()
        l3_input = self.l2(l2_input).relu()

        y = self.l3(l3_input)

        return y


# class Actor(torch.nn.Module):
#     # def __init__(self,
#     #              states_dim: int,
#     #              hidden_units: int,
#     #              action_dim: int):
#     #     super().__init__()
#     #     self.lin1 = nn.Linear(states_dim, hidden_units)
#     #     self.lin2 = nn.Linear(hidden_units, action_dim)
#     #
#     # def forward(self, state: torch.Tensor) -> torch.Tensor:
#     #     layer_2_input = self.lin1(state).relu()
#     #     y = self.lin2(layer_2_input)
#     #
#     #     return y
#
#     def __init__(self, state_dim, action_dim, max_action):
#         super().__init__()
#
#         self.l1 = nn.Linear(state_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, action_dim)
#
#         self.max_action = max_action
#
#     def forward(self, state):
#         x = F.relu(self.l1(state))
#         x = F.relu(self.l2(x))
#         return self.max_action * torch.tanh(self.l3(x))
#
#
# class Critic(torch.nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#
#         # Q1 architecture
#         self.l1 = nn.Linear(state_dim + action_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, 1)
#
#         # Q2 architecture
#         self.l4 = nn.Linear(state_dim + action_dim, 256)
#         self.l5 = nn.Linear(256, 256)
#         self.l6 = nn.Linear(256, 1)
#
#     def forward(self, state, action):
#         sa = torch.cat([state, action], 1)
#
#         q1 = F.relu(self.l1(sa))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)
