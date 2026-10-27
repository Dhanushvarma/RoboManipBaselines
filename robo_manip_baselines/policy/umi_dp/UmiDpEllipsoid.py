"""
The arm's manipulability ellipsoid as an extra UmiDp output (see TrainUmiDp --ellipsoid_label_dir).

For each action step the policy predicts the shape of the ellipsoid in the chunk's anchor EEF frame: the 6D Mandel
vector of log(shape(R_aᵀ M R_a)), where R_a is the anchor rotation (world from EEF) and shape() scales M to unit
determinant. The rollout maps it back to a unit-determinant target in world axes; the controller supplies the robot's
own volume. Labels and the SPD math come from the manipulability-learning package
(github.com/Dhanushvarma/manipulability-learning), imported only when the ellipsoid is used.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robo_manip_baselines.common.utils.PoseUtils import _get_matrix_from_rotation_6d

ELLIPSOID_DIM = 6

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


def encode_ellipsoid(ellipsoid, anchor_rot):
    """Action dims (seq_len, 6) of ellipsoids (seq_len, 3, 3) in world axes, for one anchor rotation."""
    from manipulability_learning import spd

    return spd.shape_log_vec(anchor_rot.T @ ellipsoid @ anchor_rot)


def decode_ellipsoid(action, anchor_rot):
    """Unit-determinant ellipsoids (seq_len, 3, 3) in world axes from action dims (seq_len, 6)."""
    from manipulability_learning import spd

    # Any trace the network leaves is dropped, so the target has exactly unit volume
    shape_log = spd.traceless(spd.unvec(action))
    return spd.expm(anchor_rot @ shape_log @ anchor_rot.T)


def get_ellipsoid_command(ellipsoid):
    """One end-effector's DataKey.COMMAND_EEF_ELLIPSOID entry (6,): the Mandel vector of a decoded ellipsoid."""
    from manipulability_learning import spd

    return spd.vec(ellipsoid)
