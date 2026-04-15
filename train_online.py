#! /usr/bin/env python
import os
import pickle
import shutil
import time

# Force CPU backend for portability in headless/dev environments.
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

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
from rl.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Go1Run-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 250, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 250, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', True, 'Log wandb.')
flags.DEFINE_boolean('save_video', False,
                     'Record videos with gym RecordVideo.')
flags.DEFINE_integer('video_length', 1000,
                     'Video length in frames. <=0 means one video per episode.')
flags.DEFINE_integer('video_interval', 1,
                     'Record one video every N episodes when save_video=True.')
flags.DEFINE_boolean('record_train_video', True,
                     'Record training env videos when save_video=True.')
flags.DEFINE_boolean('record_eval_video', True,
                     'Record evaluation env videos when save_video=True.')
flags.DEFINE_boolean('visualize', False,
                     'Render MuJoCo window/live plot during training.')
flags.DEFINE_enum('visualize_backend', 'auto', ['auto', 'human', 'matplotlib'],
                  'Visualization backend: auto|human|matplotlib.')
flags.DEFINE_integer('render_every', 1,
                     'Render every N env steps when visualize=True.')
flags.DEFINE_integer('render_fps', 30,
                     'Maximum on-screen render FPS when visualize=True.')
flags.DEFINE_integer('render_width', 640,
                     'Render width for matplotlib backend.')
flags.DEFINE_integer('render_height', 360,
                     'Render height for matplotlib backend.')
flags.DEFINE_boolean('render_eval', False, 'Render evaluation episodes in a window.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_boolean('randomize_ground', True,
                     'Use randomized hfield terrain in Go1 task.')
flags.DEFINE_float('terminate_pitch_roll', 12.0,
                   'Terminate episode when |roll| or |pitch| exceeds this value (deg).')
flags.DEFINE_float('terminate_body_height', -1.0,
                   'Terminate episode when body height drops below this value (m). <=0 disables.')
flags.DEFINE_integer('max_episode_steps', 400,
                     'TimeLimit cap for each episode.')
flags.DEFINE_float('run_move_speed', 0.5, 'Target forward speed command (m/s).')
flags.DEFINE_boolean('run_randomize_move_speed', False,
                     'Randomize speed command each episode.')
flags.DEFINE_float('run_move_speed_min', 0.2,
                   'Min episode speed command when randomizing.')
flags.DEFINE_float('run_move_speed_max', 1.0,
                   'Max episode speed command when randomizing.')
flags.DEFINE_float('run_speed_upper_multiplier', 2.0,
                   'Upper speed reward bound multiplier.')
flags.DEFINE_float('run_reward_margin_scale', 2.0,
                   'Speed reward margin scale. Lower values penalize standing still more.')
flags.DEFINE_float('run_yaw_penalty_weight', 0.1,
                   'Yaw-rate penalty coefficient.')
flags.DEFINE_float('run_tilt_penalty_weight', 1.0,
                   'Body tilt penalty coefficient (penalizes non-upright posture).')
flags.DEFINE_float('run_reward_scale', 10.0, 'Global reward scaling factor.')
flags.DEFINE_boolean('run_command_curriculum', True,
                     'Use speed-command curriculum when randomizing move speed.')
flags.DEFINE_integer('run_command_curriculum_steps', 30000,
                     'Steps to ramp command max speed from min to max.')
flags.DEFINE_float('run_command_curriculum_start_frac', 0.4,
                   'Initial fraction of (max-min) speed range used at step 0.')
flags.DEFINE_boolean('log_reward_terms', True,
                     'Log reward decomposition terms when available.')
flags.DEFINE_integer('reward_terms_log_interval', 200,
                     'Log instantaneous reward terms every N steps.')
flags.DEFINE_boolean('print_reward_terms', False,
                     'Print reward decomposition terms to stdout.')
flags.DEFINE_integer('print_reward_terms_interval', 200,
                     'Print reward terms every N steps.')
flags.DEFINE_boolean('print_episode_summary', True,
                     'Print episode return/length and termination reason on each episode end.')
flags.DEFINE_integer('print_episode_interval', 1,
                     'Print one episode summary every N episodes.')
flags.DEFINE_boolean('print_termination_debug', True,
                     'Print failure termination diagnostics on episode end.')
flags.DEFINE_integer('print_termination_interval', 1,
                     'Print one failure diagnostics line every N failure episodes.')
flags.DEFINE_integer('utd_ratio', 20, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


_VIS_FIG = None
_VIS_AX = None
_VIS_IM = None


def _render_human(env):
    try:
        env.unwrapped.render(mode='human')
    except Exception:
        try:
            env.render(mode='human')
        except TypeError:
            env.render()


def _render_matplotlib(env, width: int, height: int):
    global _VIS_FIG, _VIS_AX, _VIS_IM

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'matplotlib is required for --visualize_backend=matplotlib. '
            'Install it or set --visualize_backend=human.') from exc

    render_src = env.unwrapped if hasattr(env, 'unwrapped') else env
    try:
        frame = render_src.render(mode='rgb_array', width=width, height=height)
    except TypeError:
        frame = render_src.render(mode='rgb_array')
    except ValueError:
        safe_w, safe_h = min(width, 640), min(height, 360)
        frame = render_src.render(mode='rgb_array', width=safe_w, height=safe_h)

    if frame is None:
        return

    frame = np.asarray(frame)
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


def _find_task_with_attr(env, attr: str):
    to_visit = [env]
    visited = set()
    while to_visit:
        obj = to_visit.pop()
        oid = id(obj)
        if oid in visited:
            continue
        visited.add(oid)

        task = getattr(obj, 'task', None)
        if task is not None and hasattr(task, attr):
            return task

        for attr in ('env', '_env', 'unwrapped'):
            if hasattr(obj, attr):
                child = getattr(obj, attr)
                if child is not None and child is not obj:
                    to_visit.append(child)
    return None


def main(_):
    backend = FLAGS.visualize_backend
    if backend == 'auto':
        backend = 'matplotlib'

    # Matplotlib preview uses rgb_array rendering; prefer offscreen GL by default.
    if FLAGS.visualize and backend == 'matplotlib' and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
        print('[render] MUJOCO_GL not set, defaulting to egl for matplotlib preview.')

    use_wandb = FLAGS.wandb
    if FLAGS.wandb:
        wandb.init(project='go1')
        wandb.config.update(FLAGS)

    terminate_body_height = None
    if FLAGS.terminate_body_height > 0:
        terminate_body_height = FLAGS.terminate_body_height

    if FLAGS.real_robot:
        raise NotImplementedError(
            'Real-robot training path is not available in the go1-only branch.')
    else:
        from env_utils import make_mujoco_env
        env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history,
            randomize_ground=FLAGS.randomize_ground,
            terminate_pitch_roll=FLAGS.terminate_pitch_roll,
            terminate_body_height=terminate_body_height,
            max_episode_steps=FLAGS.max_episode_steps,
            move_speed=FLAGS.run_move_speed,
            randomize_move_speed=FLAGS.run_randomize_move_speed,
            move_speed_min=FLAGS.run_move_speed_min,
            move_speed_max=FLAGS.run_move_speed_max,
            speed_upper_multiplier=FLAGS.run_speed_upper_multiplier,
            reward_margin_scale=FLAGS.run_reward_margin_scale,
            yaw_penalty_weight=FLAGS.run_yaw_penalty_weight,
            tilt_penalty_weight=FLAGS.run_tilt_penalty_weight,
            reward_scale=FLAGS.run_reward_scale)

    video_interval = max(1, FLAGS.video_interval)
    train_video_requested = FLAGS.save_video and FLAGS.record_train_video
    eval_video_requested = FLAGS.save_video and FLAGS.record_eval_video
    if FLAGS.save_video and not (FLAGS.record_train_video or FLAGS.record_eval_video):
        print('Warning: save_video=True but both train/eval video flags are disabled.')

    train_save_video = train_video_requested and not FLAGS.visualize
    eval_save_video = eval_video_requested and not FLAGS.visualize
    if FLAGS.visualize and (train_video_requested or eval_video_requested):
        print(
            'Warning: visualize=True, disabling RecordVideo to avoid render conflicts.')

    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    if train_save_video:
        env = gym.wrappers.RecordVideo(
            env,
            f'videos/train_{FLAGS.action_filter_high_cut}',
            episode_trigger=lambda x: (x % video_interval == 0),
            video_length=FLAGS.video_length)
    env.seed(FLAGS.seed)

    if not FLAGS.real_robot:
        eval_env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history,
            randomize_ground=FLAGS.randomize_ground,
            terminate_pitch_roll=FLAGS.terminate_pitch_roll,
            terminate_body_height=terminate_body_height,
            max_episode_steps=FLAGS.max_episode_steps,
            move_speed=FLAGS.run_move_speed,
            randomize_move_speed=FLAGS.run_randomize_move_speed,
            move_speed_min=FLAGS.run_move_speed_min,
            move_speed_max=FLAGS.run_move_speed_max,
            speed_upper_multiplier=FLAGS.run_speed_upper_multiplier,
            reward_margin_scale=FLAGS.run_reward_margin_scale,
            yaw_penalty_weight=FLAGS.run_yaw_penalty_weight,
            tilt_penalty_weight=FLAGS.run_tilt_penalty_weight,
            reward_scale=FLAGS.run_reward_scale)
        eval_env = wrap_gym(eval_env, rescale_actions=True)
        if eval_save_video:
            eval_env = gym.wrappers.RecordVideo(
                eval_env,
                f'videos/eval_{FLAGS.action_filter_high_cut}',
                episode_trigger=lambda x: (x % video_interval == 0),
                video_length=FLAGS.video_length)
        eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)

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
    reward_task = _find_task_with_attr(env, 'reward_terms')
    termination_task = _find_task_with_attr(env, 'termination_debug')
    command_task = _find_task_with_attr(env, 'set_move_speed_range')
    if FLAGS.log_reward_terms and reward_task is None:
        print('Warning: reward_terms unavailable; reward decomposition logging disabled.')
    episode_count = 0
    timeout_count = 0
    failure_count = 0
    reward_terms_episode_sums = {}
    reward_terms_episode_count = 0
    next_render_ts = 0.0
    if FLAGS.visualize:
        _render_live(env, backend, FLAGS.render_width, FLAGS.render_height)
        next_render_ts = time.monotonic()

    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        if (FLAGS.run_randomize_move_speed and FLAGS.run_command_curriculum
                and command_task is not None):
            curriculum_steps = max(1, FLAGS.run_command_curriculum_steps)
            progress = min(1.0, float(i) / float(curriculum_steps))
            start_frac = np.clip(FLAGS.run_command_curriculum_start_frac, 0.0, 1.0)
            max_span_frac = start_frac + (1.0 - start_frac) * progress
            cur_min = FLAGS.run_move_speed_min
            cur_max = cur_min + max_span_frac * (
                FLAGS.run_move_speed_max - FLAGS.run_move_speed_min)
            command_task.set_move_speed_range(cur_min, cur_max)
        next_observation, reward, done, info = _step_env(env, action)

        reward_terms = {}
        if reward_task is not None:
            reward_terms = reward_task.reward_terms
            if reward_terms:
                for k, v in reward_terms.items():
                    reward_terms_episode_sums[k] = (
                        reward_terms_episode_sums.get(k, 0.0) + float(v))
                reward_terms_episode_count += 1

                if (FLAGS.log_reward_terms and use_wandb and
                        i % max(1, FLAGS.reward_terms_log_interval) == 0):
                    for k, v in reward_terms.items():
                        wandb.log({f'training/reward_terms_step/{k}': float(v)},
                                  step=i)
                if (FLAGS.print_reward_terms and
                        i % max(1, FLAGS.print_reward_terms_interval) == 0):
                    print(
                        '[reward_terms] '
                        f"step={i} "
                        f"cmd_range=[{reward_terms.get('move_speed_min', np.nan):.3f},"
                        f"{reward_terms.get('move_speed_max', np.nan):.3f}] "
                        f"move_speed={reward_terms.get('move_speed_cmd', np.nan):.3f} "
                        f"upright={reward_terms.get('upright_reward', np.nan):.3f} "
                        f"forward={reward_terms.get('forward_reward', np.nan):.3f} "
                        f"forward_upright={reward_terms.get('forward_upright_reward', np.nan):.3f} "
                        f"yaw_penalty={reward_terms.get('yaw_penalty', np.nan):.3f} "
                        f"tilt_penalty={reward_terms.get('tilt_penalty', np.nan):.3f} "
                        f"reward_raw={reward_terms.get('reward_raw', np.nan):.3f} "
                        f"reward={reward_terms.get('reward', np.nan):.3f} "
                        f"x_vel={reward_terms.get('x_velocity', np.nan):.3f} "
                        f"pitch_deg={reward_terms.get('pitch_deg', np.nan):.2f}",
                        flush=True)

        if FLAGS.visualize and i % max(1, FLAGS.render_every) == 0:
            now = time.monotonic()
            if now >= next_render_ts:
                _render_live(env, backend, FLAGS.render_width, FLAGS.render_height)
                next_render_ts = now + (1.0 / max(1, FLAGS.render_fps))

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
            is_timeout = bool(info.get('TimeLimit.truncated', False))
            is_failure = not is_timeout
            termination_debug = {}
            if termination_task is not None:
                termination_debug = termination_task.termination_debug
            episode_count += 1
            timeout_count += int(is_timeout)
            failure_count += int(is_failure)
            timeout_ratio = timeout_count / max(episode_count, 1)

            if (FLAGS.print_episode_summary
                    and episode_count % max(1, FLAGS.print_episode_interval) == 0):
                ep_info = info.get('episode', {})
                ep_return = ep_info.get('r', np.nan)
                ep_length = ep_info.get('l', np.nan)
                reason = 'timeout' if is_timeout else 'failure'
                print(
                    '[episode] '
                    f"count={episode_count} step={i} "
                    f"return={ep_return:.3f} length={ep_length} reason={reason}",
                    flush=True)

            if (is_failure and FLAGS.print_termination_debug
                    and failure_count % max(1, FLAGS.print_termination_interval) == 0):
                reason_code = float(termination_debug.get('reason_code', 0.0))
                if reason_code == 1.0:
                    reason = 'pitch_roll'
                elif reason_code == 2.0:
                    reason = 'body_height'
                elif reason_code == 3.0:
                    reason = 'unsafe_contact'
                else:
                    reason = 'unknown'
                print(
                    '[termination] '
                    f"episode={episode_count} reason={reason} "
                    f"roll={termination_debug.get('roll_deg', np.nan):.2f} "
                    f"pitch={termination_debug.get('pitch_deg', np.nan):.2f} "
                    f"height={termination_debug.get('body_height', np.nan):.3f} "
                    f"unsafe_contact={termination_debug.get('unsafe_contact', np.nan):.0f}",
                    flush=True)

            observation, done = _reset_env(env), False
            if use_wandb and 'episode' in info:
                decode = {'r': 'return', 'l': 'length', 't': 'time'}
                for k, v in info['episode'].items():
                    wandb.log({f'training/{decode.get(k, k)}': v}, step=i)
                wandb.log(
                    {
                        'training/episode_timeout': float(is_timeout),
                        'training/episode_failure': float(is_failure),
                        'training/timeout_ratio': timeout_ratio,
                        'training/failure_ratio': 1.0 - timeout_ratio,
                    },
                    step=i)
                if termination_debug:
                    for k in ('reason_code', 'roll_deg', 'pitch_deg',
                              'body_height', 'unsafe_contact',
                              'terminate_pitch_roll',
                              'terminate_body_height'):
                        if k in termination_debug:
                            wandb.log(
                                {f'training/termination/{k}':
                                 float(termination_debug[k])},
                                step=i)
                if (FLAGS.log_reward_terms and reward_terms_episode_count > 0):
                    for k, v in reward_terms_episode_sums.items():
                        wandb.log(
                            {f'training/reward_terms_episode/{k}':
                             float(v) / reward_terms_episode_count},
                            step=i)
            reward_terms_episode_sums = {}
            reward_terms_episode_count = 0

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if use_wandb:
                        wandb.log({f'training/{k}': v}, step=i)

        if i % FLAGS.eval_interval == 0:
            if not FLAGS.real_robot:
                eval_info = evaluate(agent,
                                     eval_env,
                                     num_episodes=FLAGS.eval_episodes,
                                     render=FLAGS.render_eval,
                                     render_mode='human')
                for k, v in eval_info.items():
                    if use_wandb:
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


if __name__ == '__main__':
    app.run(main)
