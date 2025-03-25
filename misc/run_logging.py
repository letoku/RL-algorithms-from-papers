import pickle
from misc.visualizations_helpers import save_video


def evaluate(env, agent, video_file_name, episode_duration: int = 5, render: bool = False):
    frames = []
    episode_reward, time_steps = 0, 0
    state, reward, terminated = env.reset()

    while not terminated:
        time_steps += 1
        action = agent.get_action(state=state, noise=0.0)
        state, reward, terminated = env.step(action)
        episode_reward += reward

        if render:
            camera_view = env.render()
            frames.append(camera_view)

    save_video(frames, name=video_file_name+f'__SCORE_{episode_reward}', framerate=1. / env.dt)

    return episode_reward


def log_run(episodes_scores, env, agent, video_file_name, episode: int, checkpoint: int):
    if (episode + 1) % checkpoint == 0:
        with open('scores.pkl', 'wb') as handle:
            pickle.dump(episodes_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        eval_score = evaluate(env, agent, video_file_name + f'_EPISODE_{episode + 1}', render=True)
        print(f'\n EVAL SCORE: {eval_score}')

    if (episode + 1) % (4 * checkpoint) == 0:
        with open('./agents/agent.pkl', 'wb') as handle:
            pickle.dump(agent, handle, protocol=pickle.HIGHEST_PROTOCOL)

