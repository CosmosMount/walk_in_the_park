from typing import Optional

import gym
import numpy as np
from dm_control import composer
from dmcgym import DMCGYM
from gym import spaces
from gym.wrappers import FlattenObservation

from filter import ActionFilterWrapper
from sim.robots import Go1
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


def make_env(task_name: str,
             control_frequency: int = 33,
             randomize_ground: bool = True,
             action_history: int = 1,
             terminate_pitch_roll: Optional[float] = 12.0,
             terminate_body_height: Optional[float] = None,
             move_speed: float = 0.5,
             randomize_move_speed: bool = False,
             move_speed_min: float = 0.2,
             move_speed_max: float = 1.0,
             speed_upper_multiplier: float = 2.0,
             reward_margin_scale: float = 2.0,
             yaw_penalty_weight: float = 0.1,
             tilt_penalty_weight: float = 1.0,
             reward_scale: float = 10.0):
    robot = Go1(action_history=action_history)
    # robot.kd = 5

    if task_name == 'Go1Run-v0':
        task = Run(robot,
                   control_timestep=round(1.0 / control_frequency, 3),
                   randomize_ground=randomize_ground,
                   terminate_pitch_roll=terminate_pitch_roll,
                   terminate_body_height=terminate_body_height,
                   move_speed=move_speed,
                   randomize_move_speed=randomize_move_speed,
                   move_speed_min=move_speed_min,
                   move_speed_max=move_speed_max,
                   speed_upper_multiplier=speed_upper_multiplier,
                   reward_margin_scale=reward_margin_scale,
                   yaw_penalty_weight=yaw_penalty_weight,
                   tilt_penalty_weight=tilt_penalty_weight,
                   reward_scale=reward_scale)
    else:
        raise NotImplementedError(f'Unsupported task_name: {task_name}')

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
                    randomize_ground: bool = True,
                    terminate_pitch_roll: Optional[float] = 12.0,
                    terminate_body_height: Optional[float] = None,
                    max_episode_steps: int = 400,
                    move_speed: float = 0.5,
                    randomize_move_speed: bool = False,
                    move_speed_min: float = 0.2,
                    move_speed_max: float = 1.0,
                    speed_upper_multiplier: float = 2.0,
                    reward_margin_scale: float = 2.0,
                    yaw_penalty_weight: float = 0.1,
                    tilt_penalty_weight: float = 1.0,
                    reward_scale: float = 10.0) -> gym.Env:
    env = make_env(env_name,
                   control_frequency=control_frequency,
                   action_history=action_history,
                   randomize_ground=randomize_ground,
                   terminate_pitch_roll=terminate_pitch_roll,
                   terminate_body_height=terminate_body_height,
                   move_speed=move_speed,
                   randomize_move_speed=randomize_move_speed,
                   move_speed_min=move_speed_min,
                   move_speed_max=move_speed_max,
                   speed_upper_multiplier=speed_upper_multiplier,
                   reward_margin_scale=reward_margin_scale,
                   yaw_penalty_weight=yaw_penalty_weight,
                   tilt_penalty_weight=tilt_penalty_weight,
                   reward_scale=reward_scale)

    env = gym.wrappers.TimeLimit(env, max_episode_steps)

    env = gym.wrappers.ClipAction(env)

    if action_filter_high_cut is not None:
        env = ActionFilterWrapper(env, highcut=action_filter_high_cut)

    if clip_actions:
        action_offset = np.asarray(Go1.action_offset())
        init_qpos = np.asarray(Go1.init_qpos())
        if env.action_space.shape[0] == 12:
            env = ClipAction(env, init_qpos - action_offset,
                             init_qpos + action_offset)
        else:
            env = ClipAction(
                env, np.concatenate([init_qpos - action_offset, [-1.0]]),
                np.concatenate([init_qpos + action_offset, [1.0]]))

    return env
