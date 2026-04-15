import os
from collections import deque
from functools import cached_property
from typing import Optional

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.utils.transformations import quat_to_euler
from dm_env import specs

_GO1_XML_PATH = os.path.join(os.path.dirname(__file__), 'go1', 'go1.xml')


class Go1Observables(base.WalkerObservables):

    @composer.observable
    def joints_vel(self):
        return observable.MJCFFeature('qvel', self._entity.observable_joints)

    @composer.observable
    def prev_action(self):
        return observable.Generic(lambda _: self._entity.prev_action)

    @property
    def proprioception(self):
        return ([self.joints_pos, self.joints_vel] +
                self._collect_from_attachments('proprioception'))

    @composer.observable
    def sensors_velocimeter(self):
        return observable.Generic(
            lambda physics: self._entity.get_velocity(physics))

    @property
    def kinematic_sensors(self):
        return ([
            self.sensors_gyro, self.sensors_velocimeter, self.sensors_framequat
        ] + self._collect_from_attachments('kinematic_sensors'))


class Go1(base.Walker):
    _XML_PATH = _GO1_XML_PATH

    # Joint-space action center and clipping span, matching the original code style.
    _INIT_QPOS = np.asarray([0.05, 0.9, -1.8] * 4)
    _QPOS_OFFSET = np.asarray([0.09, 0.20, 0.20] * 4)

    # Go1 joint ranges (abduction, hip, knee) x 4 legs.
    _ACTION_MIN = np.asarray([-0.863, -0.686, -2.818] * 4)
    _ACTION_MAX = np.asarray([0.863, 4.501, -0.888] * 4)

    # Go1 motor torque limits (hip/thigh: 23.7, knee: 35.55) x 4 legs.
    _CTRL_MIN = np.asarray([-23.7, -23.7, -35.55] * 4)
    _CTRL_MAX = np.asarray([23.7, 23.7, 35.55] * 4)

    _ROOT_BODY_NAME = 'trunk'
    _ROOT_HEIGHT = 0.27
    _IMU_SITE_NAME = 'imu'
    _DEFAULT_KP = 25
    _DEFAULT_KD = 9

    @staticmethod
    def _get_env_float(key: str, default: float) -> float:
        value = os.getenv(key)
        if value is None or value == '':
            return float(default)
        return float(value)

    @classmethod
    def init_qpos(cls) -> np.ndarray:
        text = os.getenv('GO1_INIT_QPOS')
        if text:
            vals = [float(v.strip()) for v in text.split(',') if v.strip()]
            if len(vals) != 3:
                raise ValueError('GO1_INIT_QPOS must provide 3 comma-separated values.')
            return np.asarray(vals * 4)
        return cls._INIT_QPOS.copy()

    @classmethod
    def action_offset(cls) -> np.ndarray:
        text = os.getenv('GO1_QPOS_OFFSET')
        if text:
            vals = [float(v.strip()) for v in text.split(',') if v.strip()]
            if len(vals) != 3:
                raise ValueError(
                    'GO1_QPOS_OFFSET must provide 3 comma-separated values.')
            return np.asarray(vals * 4)

        scale = cls._get_env_float('GO1_QPOS_OFFSET_SCALE', 1.0)
        return cls._QPOS_OFFSET.copy() * scale

    def _build(self,
               name: Optional[str] = None,
               action_history: int = 1,
               learn_kd: bool = False):
        self._mjcf_root = mjcf.from_path(self._XML_PATH)
        if name:
            self._mjcf_root.model = name

        self._root_body = self._mjcf_root.find('body', self._ROOT_BODY_NAME)
        if self._root_body is None:
            raise ValueError(
                f'Root body "{self._ROOT_BODY_NAME}" not found in {self._XML_PATH}.')
        self._root_body.pos[-1] = self._ROOT_HEIGHT

        self._actuators = self.mjcf_model.find_all('actuator')
        self._joints = [actuator.joint for actuator in self._actuators]

        assert len(self._joints) == len(self._actuators)

        self._ensure_default_sensors()

        # Keep PD-to-torque control.
        self._init_qpos = self.init_qpos()
        self.kp = self._get_env_float('GO1_PD_KP', self._DEFAULT_KP)
        if learn_kd:
            self.kd = None
        else:
            self.kd = self._get_env_float('GO1_PD_KD', self._DEFAULT_KD)

        self._torque_scale = self._get_env_float('GO1_TORQUE_SCALE', 1.0)

        self._prev_actions = deque(maxlen=action_history)
        self.initialize_episode_mjcf(None)

    def _get_or_create_imu_site(self):
        imu_site = self._mjcf_root.find('site', self._IMU_SITE_NAME)
        if imu_site is None:
            imu_site = self._root_body.add('site',
                                           name=self._IMU_SITE_NAME,
                                           pos=[0.0, 0.0, 0.0])
        return imu_site

    def _has_named_sensor(self, sensor_name):
        return self._mjcf_root.find('sensor', sensor_name) is not None

    def _ensure_default_sensors(self):
        imu_site = self._get_or_create_imu_site()
        sensor_root = self._mjcf_root.sensor

        if not self._has_named_sensor('gyro'):
            sensor_root.add('gyro', name='gyro', site=imu_site)

        if not self._has_named_sensor('velocimeter'):
            sensor_root.add('velocimeter', name='velocimeter', site=imu_site)

        if not self._has_named_sensor('framequat'):
            sensor_root.add('framequat',
                            name='framequat',
                            objtype='site',
                            objname=imu_site.name)

    def initialize_episode_mjcf(self, random_state):
        del random_state
        self._prev_actions.clear()
        for _ in range(self._prev_actions.maxlen):
            self._prev_actions.append(self._init_qpos.copy())

    @cached_property
    def action_spec(self):
        minimum = self._ACTION_MIN.copy().tolist()
        maximum = self._ACTION_MAX.copy().tolist()

        if self.kd is None:
            minimum.append(-1.0)
            maximum.append(1.0)

        return specs.BoundedArray(
            shape=(len(minimum), ),
            dtype=np.float32,
            minimum=minimum,
            maximum=maximum,
            name='\t'.join([actuator.name for actuator in self.actuators]))

    @cached_property
    def ctrllimits(self):
        minimum = []
        maximum = []
        for idx, actuator in enumerate(self.actuators):
            if actuator.ctrlrange is not None:
                minimum.append(actuator.ctrlrange[0] * self._torque_scale)
                maximum.append(actuator.ctrlrange[1] * self._torque_scale)
            else:
                minimum.append(self._CTRL_MIN[idx] * self._torque_scale)
                maximum.append(self._CTRL_MAX[idx] * self._torque_scale)

        return minimum, maximum

    def apply_action(self, physics, desired_qpos, random_state):
        del random_state

        # Updates previous action.
        desired_qpos = np.asarray(desired_qpos)
        self._prev_actions.append(desired_qpos.copy())

        joints_bind = physics.bind(self.joints)
        qpos = joints_bind.qpos
        qvel = joints_bind.qvel

        if self.kd is None:
            min_kd = 1
            max_kd = 10

            kd = (desired_qpos[-1] + 1) / 2 * (max_kd - min_kd) + min_kd
            desired_qpos = desired_qpos[:-1]
        else:
            kd = self.kd

        action = self.kp * (desired_qpos - qpos) - kd * qvel
        minimum, maximum = self.ctrllimits
        action = np.clip(action, minimum, maximum)

        physics.bind(self.actuators).ctrl = action

    def _build_observables(self):
        return Go1Observables(self)

    @property
    def root_body(self):
        return self._root_body

    @property
    def joints(self):
        return self._joints

    @property
    def observable_joints(self):
        return self._joints

    @property
    def actuators(self):
        return self._actuators

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def prev_action(self):
        return np.concatenate(self._prev_actions)

    def get_roll_pitch_yaw(self, physics):
        quat = physics.bind(self.mjcf_model.sensor.framequat).sensordata
        return np.rad2deg(quat_to_euler(quat))

    def get_velocity(self, physics):
        velocimeter = physics.bind(self.mjcf_model.sensor.velocimeter)
        return velocimeter.sensordata
