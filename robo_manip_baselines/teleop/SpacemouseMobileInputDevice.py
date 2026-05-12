import numpy as np

from .InputDeviceBase import InputDeviceBase


class SpacemouseMobileInputDevice(InputDeviceBase):
    """Spacemouse for teleoperation input device of mobile base."""

    def __init__(
        self,
        mobile_manager,
        xy_scale=1.0,
        theta_scale=1.0,
        device_params={},
    ):
        super().__init__()

        self.mobile_manager = mobile_manager
        self.xy_scale = xy_scale
        self.theta_scale = theta_scale
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
        vel = np.array(
            [
                -1.0 * self.xy_scale * self.state.y,
                self.xy_scale * self.state.x,
                -2.0 * self.theta_scale * self.state.yaw,
            ]
        )

        self.mobile_manager.set_command_vel(vel)
