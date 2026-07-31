class BodyManagerBase:
    """Manager for each body component (e.g., single arm, mobile base)."""

    def __init__(self, env, body_config):
        self.env = env
        self.body_config = body_config


class BodyConfigBase:
    """Configuration  for each body component (e.g., single arm, mobile base)."""

    # Capabilities of the body, queried instead of isinstance(..., ArmConfig) so that a
    # body which is not an arm can still own joints, a gripper or an end-effector
    # (e.g. rmb_call_m's MobileManipulatorConfig, one chain of base + arm + gripper).
    HAS_ARM = False
    HAS_GRIPPER = False
    HAS_EEF = False
