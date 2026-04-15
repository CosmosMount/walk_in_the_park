from typing import Dict

import gym
import numpy as np


def evaluate(agent,
             env: gym.Env,
             num_episodes: int,
             render: bool = False,
             render_mode: str = 'human') -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
            if render:
                env.render(mode=render_mode)

    return {
        'return': np.mean(env.return_queue),
        'length': np.mean(env.length_queue)
    }
