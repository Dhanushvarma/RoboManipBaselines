"""Control-mode vocabulary for the omni-directional mobile base.

These helpers are shared by the body manager / teleop devices (framework) and
by the real-hardware envs (which translate them into TriOrb API calls). They
are intentionally hardware-agnostic: the mapping from a mode to a concrete
robot command lives in the env, while the mode *names*, their data keys, and
their reference-frame conventions live here so every component agrees.

Four control modes (``mobile_action_type``):

    position_absolute : action = (x, y, yaw)     odometry-frame pose       [m, m, rad]
    position_relative : action = (dx, dy, dyaw)  body-frame pose delta     [m, m, rad]
    velocity_body     : action = (vx, vy, vw)    body-frame velocity       [m/s, m/s, rad/s]
    velocity_world    : action = (vx, vy, vw)    odometry-frame velocity   [m/s, m/s, rad/s]

Notes
-----
- Position modes carry an absolute/relative distinction because a pose has a
  fixed origin to reference; the framework derives the matching ``*_REL`` data
  key from consecutive absolute samples (see ``DataManager.calc_rel_data``).
- Velocity is a rate, so it has no absolute/relative distinction in value; the
  only meaningful qualifier is the reference *frame* (body vs world). Both
  velocity modes therefore share the single ``COMMAND_MOBILE_OMNI_VEL`` key, and
  the frame is recorded out-of-band (see ``mobile_vel_frame`` / env metadata).
"""

from ..data.DataKey import DataKey

POSITION_ABSOLUTE = "position_absolute"
POSITION_RELATIVE = "position_relative"
VELOCITY_BODY = "velocity_body"
VELOCITY_WORLD = "velocity_world"

#: All valid mobile_action_type values, in a stable order.
MOBILE_ACTION_TYPES = (
    POSITION_ABSOLUTE,
    POSITION_RELATIVE,
    VELOCITY_BODY,
    VELOCITY_WORLD,
)


def validate_mobile_action_type(mobile_action_type):
    """Return the mobile_action_type if valid, else raise ValueError."""
    if mobile_action_type not in MOBILE_ACTION_TYPES:
        raise ValueError(
            f"mobile_action_type must be one of {MOBILE_ACTION_TYPES}, "
            f"got: {mobile_action_type!r}"
        )
    return mobile_action_type


def is_velocity_mode(mobile_action_type):
    """True for velocity_body / velocity_world."""
    return mobile_action_type in (VELOCITY_BODY, VELOCITY_WORLD)


def is_position_mode(mobile_action_type):
    """True for position_absolute / position_relative."""
    return mobile_action_type in (POSITION_ABSOLUTE, POSITION_RELATIVE)


def mobile_command_key(mobile_action_type):
    """The COMMAND_* data key the env should step on for this mode.

    position_absolute -> COMMAND_MOBILE_OMNI_POS
    position_relative -> COMMAND_MOBILE_OMNI_POS_REL
    velocity_*        -> COMMAND_MOBILE_OMNI_VEL  (shared by both frames)
    """
    mobile_action_type = validate_mobile_action_type(mobile_action_type)
    if mobile_action_type == POSITION_ABSOLUTE:
        return DataKey.COMMAND_MOBILE_OMNI_POS
    elif mobile_action_type == POSITION_RELATIVE:
        return DataKey.COMMAND_MOBILE_OMNI_POS_REL
    else:
        return DataKey.COMMAND_MOBILE_OMNI_VEL


def mobile_vel_frame(mobile_action_type):
    """Reference frame of the mobile velocity command, or None if not applicable.

    Returns 'world' for velocity_world, 'body' for velocity_body, and None for
    the position modes (which command no velocity, so the notion of a velocity
    frame is meaningless there). For velocity modes the frame is recorded in the
    episode metadata so the velocity can be interpreted unambiguously.
    """
    mobile_action_type = validate_mobile_action_type(mobile_action_type)
    if mobile_action_type == VELOCITY_WORLD:
        return "world"
    elif mobile_action_type == VELOCITY_BODY:
        return "body"
    else:
        return None
