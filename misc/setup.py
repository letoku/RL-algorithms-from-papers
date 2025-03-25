import torch

from algorithms.td3 import TD3Agent
from algorithms.ddpg import DDPG
from algorithms.greedy import Greedy


def setup(action_spec, observation_spec):
    """parametrization of learning"""
    MEMORY_SIZE = 3000000
    OBSERVATION_DIM = 0
    for _, v in observation_spec.items():
        current_feature_dim = 1
        for dim_i in v.shape:
            current_feature_dim *= dim_i
        OBSERVATION_DIM += current_feature_dim
    # OBSERVATION_DIM = sum([v.shape[0] for k, v in observation_spec.items()])
    ACTION_DIM = action_spec.shape[0]
    ACTION_MIN = action_spec.minimum[0]
    ACTION_MAX = action_spec.maximum[0]
    NOISE_SIGMA = 0.2
    NOISE_CLIP = 0.5
    HIDDEN_UNITS = 800
    Q_POLYAK = 0.995
    PI_POLYAK = 0.995
    POLYAK_COEFFICIENT = 0.995
    GAMMA = 0.99
    BATCH_SIZE = 256
    Q_LEARNING_RATE = 1e-5
    PI_LEARNING_RATE = 1e-5
    POLICY_UPDATE_DELAY = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPG(memory_size=MEMORY_SIZE, states_dim=OBSERVATION_DIM, actions_dim=ACTION_DIM, action_min=ACTION_MIN,
                      action_max=ACTION_MAX, polyak=POLYAK_COEFFICIENT, gamma=GAMMA, batch_size=BATCH_SIZE,
                      q_learning_rate=Q_LEARNING_RATE, pi_learning_rate=PI_LEARNING_RATE, noise_sigma=NOISE_SIGMA,
                      network_hidden_units=HIDDEN_UNITS, device=device)

    # agent = TD3Agent(memory_size=MEMORY_SIZE, states_dim=OBSERVATION_DIM, actions_dim=ACTION_DIM, action_min=ACTION_MIN,
    #                 action_max=ACTION_MAX, q_polyak=Q_POLYAK, policy_polyak=PI_POLYAK, gamma=GAMMA, batch_size=BATCH_SIZE,
    #                  q_learning_rate=Q_LEARNING_RATE, pi_learning_rate=PI_LEARNING_RATE, noise=NOISE_SIGMA,
    #                  target_noise=NOISE_SIGMA, q_hidden_units=HIDDEN_UNITS, policy_hidden_units=HIDDEN_UNITS,
    #                  device=device,
    #                  noise_clip=NOISE_CLIP,
    #                  policy_update_delay=POLICY_UPDATE_DELAY)

    # agent = Greedy(states_dim=OBSERVATION_DIM, actions_dim=ACTION_DIM, action_min=ACTION_MIN, action_max=ACTION_MAX)

    return agent, device
