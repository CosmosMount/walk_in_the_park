from typing import Optional

import gym
import numpy as np
from dm_control import composer
from dmcgym import DMCGYM
from gym import spaces
from gym.wrappers import FlattenObservation

import sim
from filter import ActionFilterWrapper
from sim.robots import A1, Go1
from sim.tasks import Run


class ClipAction(gym.ActionWrapper):

    def __init__(self, env, min_action, max_action):
        super().__init__(env)

        min_action = np.asarray(min_action)
        max_action = np.asarray(max_action)

        min_action = min_action + np.zeros(env.action_space.shape,
                                           dtype=env.action_space.dtype)

        max_action = max_action + np.zeros(env.action_space.shape,
                                           dtype=env.action_space.dtype)

        min_action = np.maximum(min_action, env.action_space.low)
        max_action = np.minimum(max_action, env.action_space.high)

        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


ROBOT_CONFIGS = {
    'A1': {
        'robot_class': A1,
        'init_qpos': A1._INIT_QPOS,
        'action_offset': np.asarray([0.2, 0.4, 0.4] * 4),
    },
    'Go1': {
        'robot_class': Go1,
        'init_qpos': Go1._INIT_QPOS,
        'action_offset': np.asarray([0.2, 0.4, 0.4] * 4),
    },
}


def _resolve_task_name(task_name: str) -> str:
    if task_name == 'run':
        return task_name

    if task_name.endswith('Run-v0'):
        return 'run'

    raise NotImplementedError(
        f"Unsupported env_name/task_name '{task_name}'. Expected 'run' or a '*Run-v0' Gym id."
    )


def make_env(task_name: str,
             control_frequency: int = 33,
             randomize_ground: bool = True,
             action_history: int = 1,
             robot_class=None):
    if robot_class is None:
        robot_class = A1

    robot = robot_class(action_history=action_history)

    task_name = _resolve_task_name(task_name)

    if task_name == 'run':
        task = Run(robot,
                   control_timestep=round(1.0 / control_frequency, 3),
                   randomize_ground=randomize_ground)
    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)

    env = DMCGYM(env)
    env = FlattenObservation(env)

    return env


make_env.metadata = DMCGYM.metadata


def make_mujoco_env(env_name: str,
                    control_frequency: int,
                    clip_actions: bool = True,
                    action_filter_high_cut: Optional[float] = -1,
                    action_history: int = 1,
                    robot_class=None) -> gym.Env:
    env = make_env(env_name,
                   control_frequency=control_frequency,
                   action_history=action_history,
                   robot_class=robot_class)

    env = gym.wrappers.TimeLimit(env, 400)

    env = gym.wrappers.ClipAction(env)

    if action_filter_high_cut is not None:
        env = ActionFilterWrapper(env, highcut=action_filter_high_cut)

    if clip_actions:
        if robot_class is None or robot_class == A1:
            robot_name = 'A1'
        else:
            robot_name = 'Go1'

        config = ROBOT_CONFIGS[robot_name]
        INIT_QPOS = config['init_qpos']
        ACTION_OFFSET = config['action_offset']

        if env.action_space.shape[0] == 12:
            env = ClipAction(env, INIT_QPOS - ACTION_OFFSET,
                             INIT_QPOS + ACTION_OFFSET)
        else:
            env = ClipAction(
                env, np.concatenate([INIT_QPOS - ACTION_OFFSET, [-1.0]]),
                np.concatenate([INIT_QPOS + ACTION_OFFSET, [1.0]]))

    return env
