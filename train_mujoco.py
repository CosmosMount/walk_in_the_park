#! /usr/bin/env python
import glob
import os
import pickle
import shutil
import time

# Force a safe default before importing libraries that may transitively import JAX.
# Users can still override via shell env or --jax_platform flag.
if 'JAX_PLATFORMS' not in os.environ and 'JAX_PLATFORM_NAME' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import gym
import numpy as np
import tqdm
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Go1Run-v0', 'Gym MuJoCo environment id.')
flags.DEFINE_string('save_dir', './saved/mujoco_Go1Run-v0',
                    'Checkpoint/replay root dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('control_frequency', 20,
                     'Control frequency for Go1Run-v0 environment.')
flags.DEFINE_float('action_filter_high_cut', None,
                   'Action filter high cut.')
flags.DEFINE_integer('action_history', 1,
                     'Action history size for Go1Run-v0 environment.')
flags.DEFINE_boolean('randomize_ground', True,
                     'Use randomized hfield terrain in Go1 task.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 250, 'Training log interval.')
flags.DEFINE_integer('eval_interval', 250, 'Evaluation/checkpoint interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of random steps before updates.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_integer('print_interval', 2000,
                     'Print heartbeat every N steps when tqdm is disabled.')
flags.DEFINE_boolean('wandb', True, 'Log to Weights & Biases.')
flags.DEFINE_integer('wandb_init_timeout', 180,
                     'Wandb init timeout in seconds.')
flags.DEFINE_boolean('wandb_fail_open', True,
                     'Continue training if wandb initialization fails.')
flags.DEFINE_boolean('wandb_retry_offline', True,
                     'If online wandb init fails, retry once in offline mode.')
flags.DEFINE_boolean('visualize', False,
                     'Render MuJoCo window/live plot during training.')
flags.DEFINE_string('visualize_backend', 'auto',
                    'Visualization backend: auto|human|matplotlib.')
flags.DEFINE_integer('render_every', 1,
                     'Render every N env steps when visualize=True.')
flags.DEFINE_integer('render_fps', 30,
                     'Maximum on-screen render FPS when visualize=True.')
flags.DEFINE_integer('render_width', 640,
                     'Render width for matplotlib backend.')
flags.DEFINE_integer('render_height', 360,
                     'Render height for matplotlib backend.')
flags.DEFINE_boolean('render_eval', False,
                     'Render evaluation episodes with human backend.')
flags.DEFINE_boolean('save_video', False,
                     'Record videos with gym RecordVideo.')
flags.DEFINE_integer('video_length', 0,
                     'Video length in frames. <=0 means one video per episode.')
flags.DEFINE_integer('video_interval', 20,
                     'Record one video every N episodes when save_video=True.')
flags.DEFINE_boolean(
    'record_train_video',
    False,
    'Record training env videos. Keep False when visualize=True to avoid render conflicts.')
flags.DEFINE_integer('utd_ratio', 20, 'Update to data ratio.')
flags.DEFINE_float(
    'go1_warmup_action_std',
    None,
    'Std-dev for Gaussian warmup actions in normalized action space when '
    'training Go1 before start_training.')
flags.DEFINE_boolean('debug_actions', False,
                     'Print Go1 action/joint debug stats during training.')
flags.DEFINE_integer('debug_interval', 200,
                     'Print debug stats every N steps when debug_actions=True.')
flags.DEFINE_string(
    'jax_platform',
    'auto',
    'JAX platform: auto|cpu|cuda. auto prefers env var; defaults to cpu fallback for stability.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def _configure_jax_platform():
    requested = FLAGS.jax_platform.lower()
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('jax_platform must be one of: auto|cpu|cuda')

    existing = os.environ.get('JAX_PLATFORMS', '').strip().lower()
    if requested == 'auto':
        if existing:
            selected = existing
            print(f'[jax] using JAX_PLATFORMS from environment: {existing}',
                  flush=True)
        else:
            selected = 'cpu'
            os.environ['JAX_PLATFORMS'] = 'cpu'
            print('[jax] JAX_PLATFORMS not set; defaulting to cpu for stability.',
                  flush=True)
    else:
        selected = requested
        os.environ['JAX_PLATFORMS'] = requested
        print(f'[jax] forcing JAX_PLATFORMS={requested}', flush=True)

    os.environ['JAX_PLATFORM_NAME'] = selected
    if selected == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    return selected


def _discover_mujoco_runtime():
    try:
        import mujoco
    except Exception as exc:
        print(f'[mujoco] python package not importable: {type(exc).__name__}: {exc}',
              flush=True)
        return

    package_dir = os.path.dirname(mujoco.__file__)
    lib_candidates = sorted(glob.glob(os.path.join(package_dir, 'libmujoco.so*')))
    if lib_candidates:
        print(f'[mujoco] using runtime: {lib_candidates[0]}', flush=True)
    else:
        print(f'[mujoco] package loaded from {package_dir}', flush=True)


def _reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        if truncated:
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info
    return out


def _make_env(env_name: str,
              seed: int,
              save_video: bool,
              video_dir: str,
              video_length: int,
              video_interval: int) -> gym.Env:
    from env_utils import make_mujoco_env
    from rl.wrappers import wrap_gym

    env = make_mujoco_env(
        env_name,
        control_frequency=FLAGS.control_frequency,
        action_filter_high_cut=FLAGS.action_filter_high_cut,
        action_history=FLAGS.action_history,
        randomize_ground=FLAGS.randomize_ground)

    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

    if save_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_dir,
            episode_trigger=lambda x: (x % max(1, video_interval) == 0),
            video_length=video_length)

    env.seed(seed)
    return env


def _render_human(env):
    try:
        env.unwrapped.render(mode='human')
    except Exception:
        try:
            env.render(mode='human')
        except TypeError:
            env.render()


_VIS_FIG = None
_VIS_AX = None
_VIS_IM = None


def _render_matplotlib(env, width: int, height: int):
    global _VIS_FIG, _VIS_AX, _VIS_IM

    import matplotlib.pyplot as plt

    try:
        frame = env.unwrapped.render(mode='rgb_array', width=width, height=height)
    except ValueError:
        # Some MuJoCo models have smaller offscreen framebuffer limits.
        safe_w, safe_h = min(width, 640), min(height, 360)
        frame = env.unwrapped.render(mode='rgb_array', width=safe_w, height=safe_h)

    if _VIS_FIG is None:
        plt.ion()
        _VIS_FIG, _VIS_AX = plt.subplots(num='MuJoCo Live View')
        _VIS_IM = _VIS_AX.imshow(frame)
        _VIS_AX.axis('off')
        _VIS_FIG.tight_layout(pad=0)
    else:
        _VIS_IM.set_data(frame)

    _VIS_FIG.canvas.draw_idle()
    plt.pause(0.001)


def _render_live(env, backend: str, width: int, height: int):
    if backend == 'matplotlib':
        _render_matplotlib(env, width, height)
    else:
        _render_human(env)


def main(_):
    if FLAGS.env_name != 'Go1Run-v0':
        raise ValueError('This branch is Go1-only. Use --env_name=Go1Run-v0.')

    selected_platform = _configure_jax_platform()

    import jax
    jax.config.update('jax_platform_name', selected_platform)

    _discover_mujoco_runtime()

    # Import after platform configuration so JAX backend selection is deterministic.
    from rl.agents import SACLearner
    from rl.data import ReplayBuffer
    from rl.evaluation import evaluate

    use_wandb = FLAGS.wandb
    if use_wandb:
        try:
            wandb.init(
                project='go1',
                settings=wandb.Settings(init_timeout=FLAGS.wandb_init_timeout))
            wandb.config.update(FLAGS)
        except Exception as exc:
            if FLAGS.wandb_fail_open:
                print(f'[wandb] online init failed: {type(exc).__name__}: {exc}',
                      flush=True)

                if FLAGS.wandb_retry_offline:
                    try:
                        wandb.init(
                            project='go1',
                            mode='offline',
                            settings=wandb.Settings(
                                init_timeout=max(30, FLAGS.wandb_init_timeout)))
                        wandb.config.update(FLAGS)
                        print('[wandb] switched to offline mode.', flush=True)
                    except Exception as exc_offline:
                        print(
                            '[wandb] offline init failed, disabling wandb: '
                            f'{type(exc_offline).__name__}: {exc_offline}',
                            flush=True)
                        use_wandb = False
                else:
                    print('[wandb] disabling wandb sync and continuing.', flush=True)
                    use_wandb = False
            else:
                raise

    backend = FLAGS.visualize_backend.lower()
    if backend not in ('auto', 'human', 'matplotlib'):
        raise ValueError('visualize_backend must be one of: auto|human|matplotlib')
    if backend == 'auto':
        backend = 'matplotlib'

    # Video wrappers spawn ffmpeg subprocesses, which can deadlock with JAX threads on some setups.
    # Keep visualization and video recording mutually exclusive for stability.
    train_save_video = FLAGS.save_video and FLAGS.record_train_video and not FLAGS.visualize
    eval_save_video = FLAGS.save_video and not FLAGS.visualize

    env = _make_env(
        FLAGS.env_name,
        FLAGS.seed,
        train_save_video,
        os.path.join('videos', f'mujoco_train_{FLAGS.env_name}'),
        FLAGS.video_length,
        FLAGS.video_interval)

    eval_env = _make_env(
        FLAGS.env_name,
        FLAGS.seed + 42,
        eval_save_video,
        os.path.join('videos', f'mujoco_eval_{FLAGS.env_name}'),
        FLAGS.video_length,
        FLAGS.video_interval)

    if FLAGS.debug_actions:
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        print(
            '[debug] Go1 action_space bounds '
            f'min={low.min():.3f}, max={high.max():.3f}',
            flush=True)

    if FLAGS.visualize and FLAGS.save_video:
        print('Warning: disabling all RecordVideo wrappers while visualize=True to avoid MuJoCo UI hangs.')

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space, env.action_space,
                              **kwargs)

    run_dir = os.path.abspath(FLAGS.save_dir)
    chkpt_dir = os.path.join(run_dir, 'checkpoints')
    buffer_dir = os.path.join(run_dir, 'buffers')
    os.makedirs(chkpt_dir, exist_ok=True)

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    start_i = 0
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    if last_checkpoint is not None:
        restored_step = int(last_checkpoint.split('_')[-1])
        if restored_step >= FLAGS.max_steps:
            print(
                'Latest checkpoint step '
                f'({restored_step}) >= max_steps ({FLAGS.max_steps}). '
                'Starting from scratch at step 0.')
        else:
            buffer_path = os.path.join(buffer_dir, f'buffer_{restored_step}')
            try:
                agent = checkpoints.restore_checkpoint(last_checkpoint, agent)
                with open(buffer_path, 'rb') as f:
                    replay_buffer = pickle.load(f)
                start_i = restored_step
                print(
                    f'Restored checkpoint and replay buffer from step {start_i}.')
            except Exception as e:
                # Older numpy / pickle formats can make replay buffers unreadable.
                # Fallback keeps training runnable instead of crashing at startup.
                print(f'Failed to restore replay buffer from {buffer_path}: {e}')
                print('Falling back to a fresh replay buffer and restarting from step 0.')

    observation, done = _reset_env(env), False
    next_render_ts = 0.0
    if FLAGS.visualize:
        _render_live(env, backend, FLAGS.render_width, FLAGS.render_height)
        next_render_ts = time.monotonic()

    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if (not FLAGS.tqdm) and i % max(1, FLAGS.print_interval) == 0:
            print(f'[train] step={i}/{FLAGS.max_steps}, done={done}', flush=True)

        if i < FLAGS.start_training:
            if FLAGS.go1_warmup_action_std is not None and FLAGS.go1_warmup_action_std > 0:
                action = np.random.normal(
                    loc=0.0,
                    scale=FLAGS.go1_warmup_action_std,
                    size=env.action_space.shape)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                action = action.astype(env.action_space.dtype)
            else:
                action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        next_observation, reward, done, info = _step_env(env, action)

        if (FLAGS.debug_actions and i % max(1, FLAGS.debug_interval) == 0):
            act = np.asarray(action)
            obs_joint = np.asarray(observation[:12])
            next_obs_joint = np.asarray(next_observation[:12])
            print(
                '[debug] '
                f'step={i} '
                f'action[min={act.min():.3f}, max={act.max():.3f}, mean={act.mean():.3f}] '
                f'joint_pos[min={obs_joint.min():.3f}, max={obs_joint.max():.3f}] '
                f'next_joint_pos[min={next_obs_joint.min():.3f}, max={next_obs_joint.max():.3f}] '
                f'reward={float(reward):.3f}',
                flush=True)

        if FLAGS.visualize and i % max(1, FLAGS.render_every) == 0:
            now = time.monotonic()
            if now >= next_render_ts:
                _render_live(env, backend, FLAGS.render_width, FLAGS.render_height)
                next_render_ts = now + (1.0 / max(1, FLAGS.render_fps))

        mask = 1.0 if (not done or 'TimeLimit.truncated' in info) else 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation

        if done:
            observation, done = _reset_env(env), False
            if use_wandb and 'episode' in info:
                decode = {'r': 'return', 'l': 'length', 't': 'time'}
                for k, v in info['episode'].items():
                    wandb.log({f'training/{decode.get(k, k)}': v}, step=i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if use_wandb and i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes,
                                 render=FLAGS.render_eval,
                                 render_mode='human')
            if use_wandb:
                for k, v in eval_info.items():
                    wandb.log({f'evaluation/{k}': v}, step=i)
            else:
                print(f'[eval step={i}] return={eval_info["return"]:.3f}, length={eval_info["length"]:.1f}',
                      flush=True)

            checkpoints.save_checkpoint(chkpt_dir,
                                        agent,
                                        step=i + 1,
                                        keep=20,
                                        overwrite=True)

            try:
                shutil.rmtree(buffer_dir)
            except Exception:
                pass

            os.makedirs(buffer_dir, exist_ok=True)
            with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                pickle.dump(replay_buffer, f)


if __name__ == '__main__':
    app.run(main)
