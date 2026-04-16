#! /usr/bin/env python
import os
import pickle
import shutil
import sys


def _extract_flag_value(argv, name):
    prefix = f'--{name}='
    for idx, arg in enumerate(argv):
        if arg.startswith(prefix):
            return arg[len(prefix):]
        if arg == f'--{name}' and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _extract_bool_flag(argv, name):
    value = _extract_flag_value(argv, name)
    if value is not None:
        return value.lower() not in ('0', 'false', 'no')
    return f'--{name}' in argv


def _preconfigure_runtime_env(argv):
    jax_platform = _extract_flag_value(argv, 'jax_platform')
    require_gpu = _extract_bool_flag(argv, 'require_gpu')

    # JAX/Flax are imported transitively at module import time, so backend
    # configuration must happen before importing project modules.
    if jax_platform is not None and 'JAX_PLATFORMS' in os.environ:
        print('Ignoring JAX_PLATFORMS because --jax_platform was provided.')
        os.environ.pop('JAX_PLATFORMS', None)
    elif os.environ.get('JAX_PLATFORMS', '').lower() == 'gpu':
        print('Ignoring JAX_PLATFORMS=gpu because JAX 0.6 expands it to '
              'both cuda and rocm. Use --jax_platform=cuda instead.')
        os.environ.pop('JAX_PLATFORMS', None)

    if jax_platform is not None:
        os.environ['JAX_PLATFORM_NAME'] = jax_platform
    elif require_gpu:
        os.environ.setdefault('JAX_PLATFORM_NAME', 'cuda')

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


_preconfigure_runtime_env(sys.argv[1:])

import numpy as np
import tqdm

import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.visualization import RemotePreviewServer
from rl.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
flags.DEFINE_string('robot_type', 'A1', 'Robot type: A1 or Go1.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', True, 'Log wandb.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('preview_stream', False,
                     'Serve a live preview stream over HTTP during training.')
flags.DEFINE_string('preview_host', '0.0.0.0',
                    'Host interface for the live preview server.')
flags.DEFINE_integer('preview_port', 8080,
                     'Port for the live preview server.')
flags.DEFINE_integer('preview_width', 640,
                     'Rendered preview frame width.')
flags.DEFINE_integer('preview_height', 360,
                     'Rendered preview frame height.')
flags.DEFINE_string('preview_camera', 'side_camera',
                    'Camera name or id used for simulation previews.')
flags.DEFINE_integer('preview_interval', 4,
                     'Publish one preview frame every N environment steps.')
flags.DEFINE_float('preview_fps', 10.0,
                   'Maximum frame rate served by the live preview endpoint.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
flags.DEFINE_boolean('force_clean_exit', False,
                     'Force os._exit(0) after shutdown to avoid native teardown crashes.')
flags.DEFINE_string('jax_platform', None,
                    'Preferred JAX backend, e.g. "cuda" or "cpu".')
flags.DEFINE_boolean('print_jax_backend', True,
                     'Print the selected JAX backend and visible devices at startup.')
flags.DEFINE_boolean('require_gpu', False,
                     'Fail fast if JAX does not select a CUDA/ROCm backend.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    import jax

    def create_replay_buffer():
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps)
        replay_buffer.seed(FLAGS.seed)
        return replay_buffer

    try:
        backend = jax.default_backend()
        devices = ', '.join(str(device) for device in jax.devices())
    except RuntimeError as exc:
        raise RuntimeError(
            'Failed to initialize JAX. If you exported JAX_PLATFORMS=gpu, '
            'clear it and use --jax_platform=cuda instead.') from exc

    if FLAGS.print_jax_backend:
        print(f'JAX backend: {backend}; devices: {devices}')
    if FLAGS.require_gpu and backend not in ('cuda', 'rocm'):
        raise RuntimeError(
            'GPU was requested but JAX did not select a GPU backend. '
            f'Current backend: {backend}.')

    if FLAGS.wandb:
        wandb.init(project='a1')
        wandb.config.update(FLAGS)

    preview_server = None
    preview_failed = False
    if FLAGS.preview_stream:
        preview_server = RemotePreviewServer(host=FLAGS.preview_host,
                                             port=FLAGS.preview_port,
                                             stream_fps=FLAGS.preview_fps)
        preview_server.start()
        print(f'Live preview: http://{FLAGS.preview_host}:{FLAGS.preview_port}/')

    if FLAGS.real_robot:
        from real.envs.a1_env import A1Real
        env = A1Real(zero_action=np.asarray([0.05, 0.9, -1.8] * 4))
    else:
        from env_utils import make_mujoco_env
        from sim.robots import A1, Go1

        if FLAGS.robot_type == 'Go1':
            robot_class = Go1
        else:
            robot_class = A1

        env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history,
            robot_class=robot_class)

    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    if FLAGS.save_video:
        env = gym.wrappers.RecordVideo(
            env,
            f'videos/train_{FLAGS.action_filter_high_cut}',
            episode_trigger=lambda x: True)
    env.seed(FLAGS.seed)

    if not FLAGS.real_robot:
        from env_utils import make_mujoco_env
        from sim.robots import A1, Go1

        if FLAGS.robot_type == 'Go1':
            robot_class = Go1
        else:
            robot_class = A1

        eval_env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history,
            robot_class=robot_class)
        eval_env = wrap_gym(eval_env, rescale_actions=True)
        if FLAGS.save_video:
            eval_env = gym.wrappers.RecordVideo(
                eval_env,
                f'videos/eval_{FLAGS.action_filter_high_cut}',
                episode_trigger=lambda x: True)
        eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)
    preview_interval = max(1, FLAGS.preview_interval)

    def resolve_preview_camera(render_env):
        if FLAGS.real_robot:
            return FLAGS.preview_camera
        if FLAGS.preview_camera.lstrip('-').isdigit():
            return int(FLAGS.preview_camera)

        base_env = render_env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        physics = getattr(getattr(base_env, '_env', None), 'physics', None)
        if physics is None:
            return FLAGS.preview_camera

        camera_names = [
            physics.model.id2name(i, 'camera') for i in range(physics.model.ncam)
        ]
        if FLAGS.preview_camera in camera_names:
            return FLAGS.preview_camera

        suffix = f'/{FLAGS.preview_camera}'
        for name in camera_names:
            if name and name.endswith(suffix):
                print(f'Resolved preview camera {FLAGS.preview_camera!r} -> '
                      f'{name!r}')
                return name

        if camera_names:
            print(f'Preview camera {FLAGS.preview_camera!r} not found. '
                  f'Falling back to {camera_names[0]!r}.')
            return camera_names[0]

        return -1

    preview_camera = resolve_preview_camera(env)

    def publish_preview_frame(render_env, step, source='train'):
        nonlocal preview_failed
        if preview_server is None or preview_failed:
            return
        if step > 0 and step % preview_interval != 0:
            return

        try:
            if FLAGS.real_robot:
                frame = render_env.render(mode='rgb_array')
            else:
                frame = render_env.render(mode='rgb_array',
                                          height=FLAGS.preview_height,
                                          width=FLAGS.preview_width,
                                          camera_id=preview_camera)
            preview_server.update_frame(frame, step=step, source=source)
        except Exception as exc:
            preview_failed = True
            print(f'Disabling live preview after render failure: {exc}')

    save_dir = os.path.abspath(FLAGS.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    chkpt_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = os.path.join(save_dir, 'buffers')

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = create_replay_buffer()
    else:
        try:
            start_i = int(last_checkpoint.split('_')[-1])

            agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

            with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
                replay_buffer = pickle.load(f)
        except Exception as exc:
            print(f'Failed to restore checkpoint state: {exc}. Starting fresh.')
            start_i = 0
            replay_buffer = create_replay_buffer()

    observation, done = env.reset(), False
    publish_preview_frame(env, start_i)
    try:
        for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                           smoothing=0.1,
                           disable=not FLAGS.tqdm):
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action, agent = agent.sample_actions(observation)
            next_observation, reward, done, info = env.step(action)
            publish_preview_frame(env, i + 1)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(
                dict(observations=observation,
                     actions=action,
                     rewards=reward,
                     masks=mask,
                     dones=done,
                     next_observations=next_observation))
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                publish_preview_frame(env, i + 1)
                for k, v in info['episode'].items():
                    decode = {'r': 'return', 'l': 'length', 't': 'time'}
                    if FLAGS.wandb:
                        wandb.log({f'training/{decode[k]}': v}, step=i)

            if i >= FLAGS.start_training:
                batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                agent, update_info = agent.update(batch, FLAGS.utd_ratio)

                if (i + 1) % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        if FLAGS.wandb:
                            wandb.log({f'training/{k}': v}, step=i)

            if (i + 1) % FLAGS.eval_interval == 0:
                if not FLAGS.real_robot:
                    eval_info = evaluate(agent,
                                         eval_env,
                                         num_episodes=FLAGS.eval_episodes)
                    for k, v in eval_info.items():
                        if FLAGS.wandb:
                            wandb.log({f'evaluation/{k}': v}, step=i)

                checkpoints.save_checkpoint(chkpt_dir,
                                            agent,
                                            step=i + 1,
                                            keep=20,
                                            overwrite=True)

                try:
                    shutil.rmtree(buffer_dir)
                except:
                    pass

                os.makedirs(buffer_dir, exist_ok=True)
                with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                    pickle.dump(replay_buffer, f)
    finally:
        env.close()
        if not FLAGS.real_robot:
            eval_env.close()
        if preview_server is not None:
            preview_server.close()
        if FLAGS.wandb:
            wandb.finish()
        if FLAGS.force_clean_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == '__main__':
    app.run(main)
