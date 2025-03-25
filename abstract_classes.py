from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Inferred from dm_control's suite tasks
    To check implementations, read through the code in Simulation.
    The suite doesn't inherit from this class.
    """

    @abstractmethod
    def action_spec(self):
        pass

    @abstractmethod
    def observation_spec(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def random_action(self):
        pass

    @property
    @abstractmethod
    def simulation_time(self):
        pass

    @property
    @abstractmethod
    def dt(self):
        pass

    @abstractmethod
    def render(self):
        pass


class Agent(ABC):
    """
    Usage:
    1. initialise
    2. get_action (inputs and outputs are normed and denormed)
    3. push to add experience to memory (include arguments if necessary)
    4. update to train model
    """

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def push(self, transition):
        pass

    @abstractmethod
    def update(self, batch_size):
        pass


class AbstractSimulation(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def show_simulation(self):
        pass

    @abstractmethod
    def get_action(self, state, t):
        pass

    @abstractmethod
    def modify_obs(self, obs):
        pass

    @abstractmethod
    def modify_action(self, action, state, t):
        pass
