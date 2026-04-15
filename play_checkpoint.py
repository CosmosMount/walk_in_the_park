#! /usr/bin/env python
import os

# Force CPU backend for portability in headless/dev environments.
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import gym
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags

from rl.agents import SACLearner
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Go1Run-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Checkpoint root dir used in training.')
flags.DEFINE_string('checkpoint_path', '',
                    'Optional explicit checkpoint path. If empty, use latest in save_dir/checkpoints.')
flags.DEFINE_integer('eval_episodes', 3, 'Number of episodes for playback/eval.')
flags.DEFINE_boolean('render', True, 'Render in a window during playback.')
flags.DEFINE_boolean('save_video', False, 'Record playback video.')
flags.DEFINE_string('video_dir', 'videos/playback', 'Video output dir when save_video=True.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the hyperparameter configuration for building the agent.',
    lock_config=False)


def main(_):
    from env_utils import make_mujoco_env

    env = make_mujoco_env(
        FLAGS.env_name,
        control_frequency=FLAGS.control_frequency,
        action_filter_high_cut=FLAGS.action_filter_high_cut,
        action_history=FLAGS.action_history)
    env = wrap_gym(env, rescale_actions=True)
    env.seed(FLAGS.seed)

    if FLAGS.save_video:
        os.makedirs(FLAGS.video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            FLAGS.video_dir,
            episode_trigger=lambda _: True)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)

    if FLAGS.checkpoint_path:
        checkpoint_path = FLAGS.checkpoint_path
    else:
        checkpoint_dir = os.path.join(os.path.abspath(FLAGS.save_dir), 'checkpoints')
        checkpoint_path = checkpoints.latest_checkpoint(checkpoint_dir)

    if not checkpoint_path:
        raise ValueError(
            'No checkpoint found. Pass --checkpoint_path or ensure --save_dir has checkpoints.')

    print(f'Loading checkpoint: {checkpoint_path}')
    agent = checkpoints.restore_checkpoint(checkpoint_path, agent)

    info = evaluate(agent,
                    env,
                    num_episodes=FLAGS.eval_episodes,
                    render=FLAGS.render,
                    render_mode='human')
    print(f"return={float(info['return']):.3f}, length={float(info['length']):.3f}")


if __name__ == '__main__':
    app.run(main)
