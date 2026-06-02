import numpy as np

from robo_manip_baselines.common import (
    is_velocity_mode,
    validate_mobile_action_type,
    POSITION_ABSOLUTE,
)

from .InputDeviceBase import InputDeviceBase


class RealSpacemouseMobileInputDevice(InputDeviceBase):
    """Spacemouse teleop for the *position/velocity-controlled* real mobile base.

    This is the real-hardware counterpart of ``SpacemouseMobileInputDevice``
    (which emits a velocity via ``set_command_vel`` and is still used by the
    velocity-controlled sim envs). It supports all four ``mobile_action_type``
    modes and writes the matching command channel on the ``MobileOmniManager``:

      - position_absolute : per-step increment accumulated into the absolute
        target pose in the odometry/origin frame (``target += delta``) ->
        ``set_command_pos``. "Forward" means +x in the odometry frame.
      - position_relative : per-step body-frame delta -> ``set_command_pos_rel``;
        the manager composes it in the body frame, so "forward" follows heading.
      - velocity_body / velocity_world : the deflection maps to a velocity
        (vx, vy, vw) -> ``set_command_vel``. The body/world distinction is
        applied by the env (set_vel_relative vs set_vel_absolute); the device
        just emits the velocity vector.

    Two independent scale sets are used because the channels have different
    units:
      - ``pos_xy_scale`` / ``pos_theta_scale``: per-step pose increments
        [m] / [rad] (small, like ``KeyboardInputDevice.pos_scale``).
      - ``vel_xy_scale`` / ``vel_theta_scale``: commanded velocity [m/s] / [rad/s]
        (the env additionally clips these to ``mobile_max_vel``).

    Axis mapping (shared by all modes): forward (-y) -> +x, right (x) -> +y,
    twist (yaw) -> -yaw.
    """

    def __init__(
        self,
        mobile_manager,
        mobile_action_type="position_relative",
        pos_xy_scale=1e-2,
        pos_theta_scale=2e-2,
        vel_xy_scale=0.2,
        vel_theta_scale=0.5,
        device_params={},
    ):
        super().__init__()

        self.mobile_action_type = validate_mobile_action_type(mobile_action_type)

        self.mobile_manager = mobile_manager
        self.pos_xy_scale = pos_xy_scale
        self.pos_theta_scale = pos_theta_scale
        self.vel_xy_scale = vel_xy_scale
        self.vel_theta_scale = vel_theta_scale
        self.device_params = device_params

    def connect(self):
        if self.connected:
            return

        self.connected = True

        import pyspacemouse

        self.spacemouse = pyspacemouse.open(**self.device_params)

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        # Empirically, you can call read repeatedly to get the latest device state
        for i in range(10):
            self.state = self.spacemouse.read()

    def set_command_data(self):
        # Normalized deflection with the shared axis mapping (each component in
        # roughly [-1, 1]).
        raw = np.array([-self.state.y, self.state.x, -self.state.yaw])

        if is_velocity_mode(self.mobile_action_type):
            vel = np.array(
                [
                    self.vel_xy_scale * raw[0],
                    self.vel_xy_scale * raw[1],
                    self.vel_theta_scale * raw[2],
                ]
            )
            self.mobile_manager.set_command_vel(vel)
            return

        # Position modes: scale the deflection into a small per-step increment.
        delta = np.array(
            [
                self.pos_xy_scale * raw[0],
                self.pos_xy_scale * raw[1],
                self.pos_theta_scale * raw[2],
            ]
        )

        if self.mobile_action_type == POSITION_ABSOLUTE:
            # Accumulate in the odometry/origin frame and command the absolute pose.
            target = np.asarray(
                self.mobile_manager.get_command_pos(), dtype=np.float64
            ).copy()
            target += delta
            self.mobile_manager.set_command_pos(target)
        else:
            # position_relative: body-frame delta composed by the manager.
            self.mobile_manager.set_command_pos_rel(delta)
