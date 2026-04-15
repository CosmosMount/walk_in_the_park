from typing import Dict, Optional, Tuple

import dm_control.utils.transformations as tr
import numpy as np
from dm_control import composer
from dm_control.locomotion import arenas
from dm_control.utils import rewards

from sim.arenas import HField
from sim.tasks.utils import _find_non_contacting_height

DEFAULT_CONTROL_TIMESTEP = 0.03
DEFAULT_PHYSICS_TIMESTEP = 0.001


def get_run_reward_terms(x_velocity: float,
                         move_speed: float,
                         cos_pitch: float,
                         cos_roll: float,
                         terminate_pitch_roll_deg: Optional[float],
                         dyaw: float,
                         speed_upper_multiplier: float = 2.0,
                         reward_margin_scale: float = 2.0,
                         yaw_penalty_weight: float = 0.1,
                         tilt_penalty_weight: float = 1.0,
                         reward_scale: float = 10.0) -> Dict[str, float]:
    # NOTE: legacy implementation used reward_margin_scale=2.0, which gives
    # non-zero forward reward even near zero speed. Keeping it configurable
    # lets us diagnose and fix "sit and don't move" local optima.
    speed_signal = cos_pitch * x_velocity
    forward_reward = rewards.tolerance(speed_signal,
                                       bounds=(move_speed,
                                               speed_upper_multiplier *
                                               move_speed),
                                       margin=reward_margin_scale * move_speed,
                                       value_at_margin=0,
                                       sigmoid='linear')
    cos_pitch_cos_roll = cos_pitch * cos_roll
    if terminate_pitch_roll_deg is not None:
        termination = np.cos(np.deg2rad(terminate_pitch_roll_deg))
        # Match A1-style upright shaping: as body tilt approaches the
        # safety threshold, forward reward gets suppressed.
        upright_reward = rewards.tolerance(cos_pitch_cos_roll,
                                           bounds=(termination, float('inf')),
                                           sigmoid='linear',
                                           margin=termination + 1,
                                           value_at_margin=0)
    else:
        upright_reward = 1.0
    forward_upright_reward = upright_reward * forward_reward
    yaw_penalty = yaw_penalty_weight * np.abs(dyaw)
    # Continuous tilt penalty helps avoid the "lean-forward then fall" mode.
    tilt_penalty = tilt_penalty_weight * max(0.0, 1.0 - cos_pitch_cos_roll)
    reward_raw = forward_upright_reward - yaw_penalty - tilt_penalty
    reward = reward_scale * reward_raw

    return dict(
        x_velocity=float(x_velocity),
        speed_signal=float(speed_signal),
        move_speed_cmd=float(move_speed),
        cos_pitch=float(cos_pitch),
        cos_roll=float(cos_roll),
        cos_pitch_cos_roll=float(cos_pitch_cos_roll),
        upright_reward=float(upright_reward),
        dyaw=float(dyaw),
        forward_reward=float(forward_reward),
        forward_upright_reward=float(forward_upright_reward),
        yaw_penalty=float(yaw_penalty),
        tilt_penalty=float(tilt_penalty),
        reward_raw=float(reward_raw),
        reward=float(reward),
    )


class Run(composer.Task):

    def __init__(self,
                 robot,
                 terminate_pitch_roll: Optional[float] = 12,
                 terminate_body_height: Optional[float] = None,
                 physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
                 floor_friction: Tuple[float] = (1, 0.005, 0.0001),
                 randomize_ground: bool = True,
                 add_velocity_to_observations: bool = True,
                 move_speed: float = 0.5,
                 randomize_move_speed: bool = False,
                 move_speed_min: float = 0.2,
                 move_speed_max: float = 1.0,
                 speed_upper_multiplier: float = 2.0,
                 reward_margin_scale: float = 2.0,
                 yaw_penalty_weight: float = 0.1,
                 tilt_penalty_weight: float = 1.0,
                 reward_scale: float = 10.0):

        self.floor_friction = floor_friction
        if randomize_ground:
            self._floor = HField(size=(10, 10))
            self._floor.mjcf_model.size.nconmax = 400
            self._floor.mjcf_model.size.njmax = 2000
        else:
            self._floor = arenas.Floor(size=(10, 10))

        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = floor_friction

        self._robot = robot
        self._floor.add_free_entity(self._robot)
        self._robot_geom_prefix = f'{self._robot.mjcf_model.model}/'
        self._trunk_body_name = f'{self._robot_geom_prefix}trunk'
        self._foot_geom_names = {
            'FR',
            'FL',
            'RR',
            'RL',
            f'{self._robot_geom_prefix}FR',
            f'{self._robot_geom_prefix}FL',
            f'{self._robot_geom_prefix}RR',
            f'{self._robot_geom_prefix}RL',
        }

        observables = (self._robot.observables.proprioception +
                       self._robot.observables.kinematic_sensors +
                       [self._robot.observables.prev_action])
        for observable in observables:
            observable.enabled = True

        if not add_velocity_to_observations:
            self._robot.observables.sensors_velocimeter.enabled = False

        if hasattr(self._floor, '_top_camera'):
            self._floor._top_camera.remove()
        self._robot.mjcf_model.worldbody.add('camera',
                                             name='side_camera',
                                             pos=[0, -1, 0.5],
                                             xyaxes=[1, 0, 0, 0, 0.342, 0.940],
                                             mode="trackcom",
                                             fovy=60.0)

        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        self._terminate_pitch_roll = terminate_pitch_roll
        self._terminate_body_height = terminate_body_height

        self._base_move_speed = move_speed
        self._randomize_move_speed = randomize_move_speed
        self._move_speed_min = move_speed_min
        self._move_speed_max = move_speed_max
        self._speed_upper_multiplier = speed_upper_multiplier
        self._reward_margin_scale = reward_margin_scale
        self._yaw_penalty_weight = yaw_penalty_weight
        self._tilt_penalty_weight = tilt_penalty_weight
        self._reward_scale = reward_scale
        self._move_speed = self._base_move_speed
        self._last_reward_terms = {}
        self._last_termination_debug = {
            'terminated': 0.0,
            'reason_code': 0.0,
            'roll_deg': np.nan,
            'pitch_deg': np.nan,
            'body_height': np.nan,
            'unsafe_contact': 0.0,
            'terminate_pitch_roll': float(self._terminate_pitch_roll)
            if self._terminate_pitch_roll is not None else np.nan,
            'terminate_body_height': float(self._terminate_body_height)
            if self._terminate_body_height is not None else np.nan,
        }

    def get_reward(self, physics):
        xmat = physics.bind(self._robot.root_body).xmat.reshape(3, 3)
        roll, pitch, _ = tr.rmat_to_euler(xmat, 'XYZ')
        velocimeter = physics.bind(self._robot.mjcf_model.sensor.velocimeter)

        gyro = physics.bind(self._robot.mjcf_model.sensor.gyro)
        terms = get_run_reward_terms(
            x_velocity=velocimeter.sensordata[0],
            move_speed=self._move_speed,
            cos_pitch=np.cos(pitch),
            cos_roll=np.cos(roll),
            terminate_pitch_roll_deg=self._terminate_pitch_roll,
            dyaw=gyro.sensordata[-1],
            speed_upper_multiplier=self._speed_upper_multiplier,
            reward_margin_scale=self._reward_margin_scale,
            yaw_penalty_weight=self._yaw_penalty_weight,
            tilt_penalty_weight=self._tilt_penalty_weight,
            reward_scale=self._reward_scale,
        )
        terms['roll_rad'] = float(roll)
        terms['pitch_rad'] = float(pitch)
        terms['roll_deg'] = float(np.rad2deg(roll))
        terms['pitch_deg'] = float(np.rad2deg(pitch))
        terms['move_speed_min'] = float(self._move_speed_min)
        terms['move_speed_max'] = float(self._move_speed_max)
        self._last_reward_terms = terms
        return terms['reward']

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

        # Terrain randomization
        if hasattr(self._floor, 'regenerate'):
            self._floor.regenerate(random_state)
            self._floor.mjcf_model.visual.map.znear = 0.00025
            self._floor.mjcf_model.visual.map.zfar = 50.

        new_friction = (random_state.uniform(low=self.floor_friction[0] - 0.25,
                                             high=self.floor_friction[0] +
                                             0.25), self.floor_friction[1],
                        self.floor_friction[2])
        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = new_friction

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._floor.initialize_episode(physics, random_state)

        self._failure_termination = False
        self._last_termination_debug = {
            'terminated': 0.0,
            'reason_code': 0.0,
            'roll_deg': np.nan,
            'pitch_deg': np.nan,
            'body_height': np.nan,
            'unsafe_contact': 0.0,
            'terminate_pitch_roll': float(self._terminate_pitch_roll)
            if self._terminate_pitch_roll is not None else np.nan,
            'terminate_body_height': float(self._terminate_body_height)
            if self._terminate_body_height is not None else np.nan,
        }
        if self._randomize_move_speed:
            self._move_speed = float(
                random_state.uniform(low=self._move_speed_min,
                                     high=self._move_speed_max))
        else:
            self._move_speed = self._base_move_speed

        # Keep spawn pose aligned with the robot's active init-qpos config.
        if hasattr(self._robot, '_init_qpos'):
            init_qpos = self._robot._init_qpos
        else:
            init_qpos = self._robot._INIT_QPOS

        _find_non_contacting_height(physics,
                                    self._robot,
                                    qpos=init_qpos)

    def before_step(self, physics, action, random_state):
        pass

    def before_substep(self, physics, action, random_state):
        self._robot.apply_action(physics, action, random_state)

    def action_spec(self, physics):
        return self._robot.action_spec

    def set_move_speed_range(self, move_speed_min: float, move_speed_max: float):
        move_speed_min = float(move_speed_min)
        move_speed_max = float(move_speed_max)
        if move_speed_max < move_speed_min:
            move_speed_min, move_speed_max = move_speed_max, move_speed_min
        self._move_speed_min = move_speed_min
        self._move_speed_max = move_speed_max

    def _is_robot_geom(self, geom_name: str) -> bool:
        return bool(geom_name) and geom_name.startswith(self._robot_geom_prefix)

    def _is_unsafe_contact(self, physics) -> bool:
        # A1 terminates on unsafe states. For Go1 MuJoCo we approximate that
        # by flagging trunk/hip collisions with non-robot geoms.
        ncon = int(physics.data.ncon)
        model = physics.model
        for i in range(ncon):
            contact = physics.data.contact[i]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            geom1_name = model.id2name(geom1_id, 'geom') or ''
            geom2_name = model.id2name(geom2_id, 'geom') or ''

            geom1_robot = self._is_robot_geom(geom1_name)
            geom2_robot = self._is_robot_geom(geom2_name)

            if geom1_robot and not geom2_robot:
                body1_name = (model.id2name(int(model.geom_bodyid[geom1_id]),
                                            'body') or '')
                if (body1_name == self._trunk_body_name
                        or body1_name.endswith('_hip')):
                    return True

            if geom2_robot and not geom1_robot:
                body2_name = (model.id2name(int(model.geom_bodyid[geom2_id]),
                                            'body') or '')
                if (body2_name == self._trunk_body_name
                        or body2_name.endswith('_hip')):
                    return True
        return False

    def after_step(self, physics, random_state):
        self._failure_termination = False
        reason_code = 0.0
        roll = np.nan
        pitch = np.nan
        body_height = np.nan
        unsafe_contact = False

        if (self._terminate_pitch_roll is not None
                or self._terminate_body_height is not None):
            roll, pitch, _ = self._robot.get_roll_pitch_yaw(physics)
            body_height = float(physics.bind(self._robot.root_body).xpos[2])

        if self._terminate_pitch_roll is not None:
            if (np.abs(roll) > self._terminate_pitch_roll
                    or np.abs(pitch) > self._terminate_pitch_roll):
                self._failure_termination = True
                reason_code = 1.0

        if (not self._failure_termination
                and self._terminate_body_height is not None
                and body_height < self._terminate_body_height):
            self._failure_termination = True
            reason_code = 2.0

        if not self._failure_termination:
            unsafe_contact = self._is_unsafe_contact(physics)
            if unsafe_contact:
                self._failure_termination = True
                reason_code = 3.0

        self._last_termination_debug = {
            'terminated': float(self._failure_termination),
            'reason_code': reason_code,
            'roll_deg': float(roll),
            'pitch_deg': float(pitch),
            'body_height': float(body_height),
            'unsafe_contact': float(unsafe_contact),
            'terminate_pitch_roll': float(self._terminate_pitch_roll)
            if self._terminate_pitch_roll is not None else np.nan,
            'terminate_body_height': float(self._terminate_body_height)
            if self._terminate_body_height is not None else np.nan,
        }

    def should_terminate_episode(self, physics):
        return self._failure_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.0
        else:
            return 1.0

    @property
    def root_entity(self):
        return self._floor

    @property
    def reward_terms(self) -> Dict[str, float]:
        # Exposed for external logging/diagnostics from the training loop.
        return dict(self._last_reward_terms)

    @property
    def termination_debug(self) -> Dict[str, float]:
        return dict(self._last_termination_debug)
