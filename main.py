import numpy as np
from dm_control import suite
from dm_env import TimeStep, StepType
from datetime import datetime
import pickle
from tqdm import tqdm

from misc.visualizations_helpers import *
from misc.setup import *
from structures.transition import *
from environment import DMControlEnvironment
from misc.run_logging import log_run, evaluate


EPISODES_NUM = 10000
EXPLORATORY_EPISODES_NUM = 1000
EPISODES_BETWEEN_SAVE = 100


# domain_name, task_name = 'cartpole', 'balance'
# domain_name, task_name = 'manipulator', 'bring_ball'
# domain_name, task_name = 'humanoid', 'run'
# domain_name, task_name = 'cheetah', 'run'
LOAD = False
LOAD_PATH = 'agent.pkl'
domain_name, task_name = 'cheetah', 'run'
random_state = np.random.RandomState(0)
env = DMControlEnvironment(domain_name, task_name, random_state)
video_file_name = domain_name + '_' + task_name + '_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

if not LOAD:
    agent, device = setup(env.action_spec(), env.observation_spec())
else:
    with open(LOAD_PATH, 'rb') as handle:
        agent = pickle.load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPLORATORY_EPISODES_NUM = 0


def set_time_dependent_hparams(episode: int, constant: bool = False, constant_value: float = 0.1) -> (bool, float):
    exploratory_phase = False if episode >= EXPLORATORY_EPISODES_NUM else True

    if exploratory_phase:
        noise = max_noise
    else:
        noise = max_noise * (1 - (episode - EXPLORATORY_EPISODES_NUM) / (EPISODES_NUM - EXPLORATORY_EPISODES_NUM))

    return exploratory_phase, noise


if __name__ == '__main__':

    episodes_scores = []
    max_noise = 0.4
    action_sel_t, env_step_t, update_t = [], [], []

    for episode in tqdm(range(EPISODES_NUM)):
        state, _, _ = env.reset()
        exploratory_phase, noise = set_time_dependent_hparams(episode)
        terminated, episode_score, steps_performed = False, 0, 0

        while not terminated:
            if exploratory_phase:
                action = env.random_action()
            else:
                action = agent.get_action(state=state, noise=noise)

            next_state, reward, terminated = env.step(action)
            agent.push(Transition(state=state, action=action, next_state=next_state, reward=reward,
                                  terminated=terminated))
            state = next_state

            if not exploratory_phase:
                agent.update()
            steps_performed += 1
            episode_score += reward

        episodes_scores.append(episode_score)
        if episode_score > 50:
            with open(f'td3_{domain_name}_good_score.pkl', 'wb') as handle:
                pickle.dump(agent, handle, protocol=pickle.HIGHEST_PROTOCOL)

        log_run(episodes_scores=episodes_scores, env=env, agent=agent, video_file_name=video_file_name,
                episode=episode, checkpoint=EPISODES_BETWEEN_SAVE)
# plot_observations(time_step=time_step, ticks=ticks, observations=observations, rewards=rewards)


def plot_training_curve(episode_scores):
    import plotly.express as px
    fig = px.line(episode_scores, title='Training curve')
    fig.show()