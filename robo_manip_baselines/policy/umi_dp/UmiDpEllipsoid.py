"""
The arm's manipulability ellipsoid in UmiDp (see TrainUmiDp --ellipsoid_label_dir): an extra output, and an input.

Shapes are unit-determinant (shape() scales M to it) and expressed in the chunk's anchor EEF axes, R_aᵀ M R_a with
R_a the anchor rotation (world from EEF). Each one enters the policy as the 6D Mandel vector of a log:

- Output, "relative" (TrainUmiDp --ellipsoid_action_repr): log(C^-1/2 M C^-1/2) for each action step, with C the
  arm's shape at the anchor step. Zero holds the current shape, and the vector's length is the affine-invariant
  distance from it, the metric of the controller's task. The rollout decodes it against the measured shape
  (DataKey.MEASURED_EEF_ELLIPSOID).
- Output, "abs": log(M), the shape itself. Checkpoints from before the choice predict this.
- Input (TrainUmiDp --ellipsoid_obs): log(C) for each observation step. A history relative to the latest step would
  always end at identity and leave the policy without the current shape.

The rollout maps the output back to a unit-determinant target in world axes; the controller supplies the robot's own
volume. Labels and the SPD math come from the manipulability-learning package
(github.com/Dhanushvarma/manipulability-learning), imported only when the ellipsoid is used.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robo_manip_baselines.common.utils.PoseUtils import _get_matrix_from_rotation_6d

ELLIPSOID_DIM = 6

# Output representations (see the module docstring), the default first
ELLIPSOID_ACTION_REPRS = ("relative", "abs")

# Largest entry difference allowed between a label file's EEF rotations and the episode's
MAX_ROT_MISMATCH = 1e-3


def get_label_path(label_dir, filename):
    """Label file of an episode: <label_dir>/<episode name>.h5, as manipulability-learning writes them."""
    return Path(label_dir) / f"{Path(filename).stem}.h5"


def load_ellipsoid_labels(label_dir, filename, measured_eef_pose):
    """
    Ellipsoids (num_frames, 3, 3) of one episode in the frame of its measured EEF pose, and the label attributes.

    `measured_eef_pose` (num_frames, 7) is the learned end-effector's (see UmiDpDataset's eef_idx). Raises if the
    label file does not have one entry per frame, or if its EEF rotations are not the episode's: a label in another
    frame would otherwise train without complaint and be wrong.
    """
    from manipulability_learning.labels import read_labels

    path = get_label_path(label_dir, filename)
    ellipsoid, eef_rot, attrs = read_labels(path)
    if len(ellipsoid) != len(measured_eef_pose):
        raise ValueError(
            f"[load_ellipsoid_labels] {path} has {len(ellipsoid)} entries, but {filename} has "
            f"{len(measured_eef_pose)} frames."
        )
    quat = np.asarray(measured_eef_pose)[:, 3:7]
    rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    mismatch = np.abs(eef_rot - rot).max()
    if mismatch > MAX_ROT_MISMATCH:
        raise ValueError(
            f"[load_ellipsoid_labels] The EEF rotations in {path} (frame '{attrs.get('frame')}') differ from "
            f"the measured EEF pose of {filename} by up to {mismatch:.3g}. The labels are in another frame."
        )
    return ellipsoid, attrs


def get_anchor_rotation(anchor_pose):
    """Rotation (world from EEF) of a 9D pose (tx, ty, tz, r6...)."""
    return _get_matrix_from_rotation_6d(np.asarray(anchor_pose)[3:9])


def get_ellipsoid_action_repr(ellipsoid_info):
    """The checkpoint's output representation."""
    return ellipsoid_info.get("action_repr", "abs")


def has_ellipsoid_obs(ellipsoid_info):
    """Whether the checkpoint takes the arm's shape as input. Checkpoints from before the choice take none."""
    return bool(ellipsoid_info.get("obs", False))


def encode_ellipsoid(ellipsoid, anchor_rot, current=None):
    """
    Policy dims (seq_len, 6) of ellipsoids (seq_len, 3, 3) in world axes, for one anchor rotation.

    With `current` (3, 3), the arm's shape at the anchor step in world axes, they are relative to it ("relative");
    without, they are the shapes themselves ("abs", and the input).
    """
    from manipulability_learning import spd

    shape = spd.shape(anchor_rot.T @ ellipsoid @ anchor_rot)
    if current is not None:
        current_inv_sqrt = spd.invsqrtm(spd.shape(anchor_rot.T @ current @ anchor_rot))
        shape = current_inv_sqrt @ shape @ current_inv_sqrt
    return spd.shape_log_vec(shape)


def decode_ellipsoid(action, anchor_rot, current=None):
    """
    Unit-determinant ellipsoids (seq_len, 3, 3) in world axes from policy dims (seq_len, 6), the inverse of
    encode_ellipsoid. For "relative", `current` is the arm's shape at the anchor step, measured at rollout.
    """
    from manipulability_learning import spd

    # Any trace the network leaves is dropped, so the target has exactly unit volume
    shape = spd.expm(spd.traceless(spd.unvec(action)))
    if current is not None:
        current_sqrt = spd.sqrtm(spd.shape(anchor_rot.T @ current @ anchor_rot))
        shape = spd.sym(current_sqrt @ shape @ current_sqrt)
    return anchor_rot @ shape @ anchor_rot.T


def get_ellipsoid_command(ellipsoid):
    """One end-effector's DataKey.COMMAND_EEF_ELLIPSOID entry (6,): the Mandel vector of a decoded ellipsoid."""
    from manipulability_learning import spd

    return spd.vec(ellipsoid)


def get_measured_ellipsoid(measured):
    """The 3x3 shape in world axes of one end-effector's DataKey.MEASURED_EEF_ELLIPSOID entry (6,)."""
    from manipulability_learning import spd

    return spd.unvec(np.asarray(measured, dtype=np.float64))
