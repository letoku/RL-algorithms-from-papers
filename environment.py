import numpy as np
from dm_control import suite
from dm_env import TimeStep, StepType

from abstract_classes import Environment


class DMControlEnvironment(Environment):
    def __init__(self, domain_name, task_name, random_state):
        self.env = suite.load(domain_name, task_name, task_kwargs={'random': random_state})

    def action_spec(self):
        return self.env.action_spec()

    def observation_spec(self):
        return self.env.observation_spec()

    def reset(self):
        return self._unwrap_time_step(self.env.reset())

    def step(self, action: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        return self._unwrap_time_step(self.env.step(action))

    def random_action(self):
        return np.random.uniform(self.action_spec().minimum, self.action_spec().maximum, size=self.action_spec().shape)

    @property
    def simulation_time(self):
        return self.env.physics.data.time

    @property
    def dt(self):
        return self.env.control_timestep()

    def render(self, camera_id=0, height=200, width=200):
        return self.env.physics.render(camera_id=camera_id, height=height, width=width)

    @staticmethod
    def _unwrap_time_step(time_step: TimeStep) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Unwrap dm TimeStep object to np arrays
        :param time_step:
        :return: next_state, reward, terminated
        """

        next_state = np.concatenate([value.reshape(-1) for _, value in time_step.observation.items()],
                                    axis=0)  # !? is it always the same order of observations in this dictionary?
        reward = time_step.reward if type(time_step.reward) is np.ndarray else np.array(time_step.reward)
        terminated = np.array(True) if time_step.step_type == StepType.LAST else np.array(False)

        return next_state.reshape(1, -1), reward, terminated
