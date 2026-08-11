import numpy as np
from scipy.spatial.transform import Rotation

# Pose representations selectable for the policy input/output
POSE_REPR_LIST = ("abs", "relative", "delta")


def _normalize_vector(vec):
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, 1e-8, None)


def _get_rotation_6d_from_matrix(rot):
    return rot[..., :2, :].reshape(*rot.shape[:-2], 6)


def _get_matrix_from_rotation_6d(rot6):
    a1 = rot6[..., 0:3]
    a2 = rot6[..., 3:6]

    b1 = _normalize_vector(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize_vector(b2)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-2)


def get_pose9_from_pose7(pose):
    """Get pose (tx, ty, tz, r6...) from pose (tx, ty, tz, qw, qx, qy, qz)."""
    pose = np.asarray(pose)
    if pose.shape[-1] % 7 != 0:
        raise ValueError(
            f"[get_pose9_from_pose7] Last dimension must be divisible by 7: {pose.shape}"
        )

    num_pose = pose.shape[-1] // 7
    pose7 = pose.reshape(*pose.shape[:-1], num_pose, 7)
    pos = pose7[..., :3]
    quat = pose7[..., 3:7]

    rot = Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
    rot = rot.reshape(*pose7.shape[:-1], 3, 3)
    rot6 = _get_rotation_6d_from_matrix(rot)
    pose9 = np.concatenate([pos, rot6], axis=-1)

    return pose9.reshape(*pose.shape[:-1], 9 * num_pose)


def get_pose7_from_pose9(pose):
    """Get pose (tx, ty, tz, qw, qx, qy, qz) from pose (tx, ty, tz, r6...)."""
    pose = np.asarray(pose)
    if pose.shape[-1] % 9 != 0:
        raise ValueError(
            f"[get_pose7_from_pose9] Last dimension must be divisible by 9: {pose.shape}"
        )

    num_pose = pose.shape[-1] // 9
    pose9 = pose.reshape(*pose.shape[:-1], num_pose, 9)
    pos = pose9[..., :3]
    rot6 = pose9[..., 3:9]

    rot = _get_matrix_from_rotation_6d(rot6)
    quat = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat()
    quat = quat.reshape(*pose9.shape[:-1], 4)[..., [3, 0, 1, 2]]
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    pose7 = np.concatenate([pos, quat], axis=-1)

    return pose7.reshape(*pose.shape[:-1], 7 * num_pose)


def _get_mat_from_pose9(pose9):
    """Get homogeneous matrix (..., 4, 4) from pose (..., 9) of (tx, ty, tz, r6...)."""
    mat = np.zeros((*pose9.shape[:-1], 4, 4), dtype=np.float64)
    mat[..., :3, :3] = _get_matrix_from_rotation_6d(pose9[..., 3:9])
    mat[..., :3, 3] = pose9[..., :3]
    mat[..., 3, 3] = 1.0

    return mat


def _get_pose9_from_mat(mat):
    """Get pose (..., 9) of (tx, ty, tz, r6...) from homogeneous matrix (..., 4, 4)."""
    return np.concatenate(
        [mat[..., :3, 3], _get_rotation_6d_from_matrix(mat[..., :3, :3])], axis=-1
    )


def convert_pose_repr(pose, anchor_pose, pose_repr, backward=False):
    """
    Convert the representation of an end-effector pose sequence.

    Poses are in the 9D policy representation (tx, ty, tz, r6...) from `get_pose9_from_pose7`. Multiple
    end-effectors are concatenated along the last axis and each is anchored independently.

    Equivalent to `pose_rep` in UMI (`diffusion_policy/common/pose_repr_util.py::convert_pose_mat_rep`),
    where `anchor_pose` is called `base_pose_mat`. Note that "delta" is not a rigid SE(3) composition:
    UMI takes the world-frame *position difference* while left-multiplying the rotation, so translation
    and rotation are decoupled. This differs from `DataKey.COMMAND_EEF_POSE_REL`, which is the body-frame
    variant `inv(T_{t-1}) @ T_t` (see `DataManager.calc_rel_data`).

    Args:
        pose (np.ndarray): Pose of shape (9 * num_pose,) or sequence of shape (seq_len, 9 * num_pose).
        anchor_pose (np.ndarray): Pose the result is expressed relative to, of shape (9 * num_pose,).
            For "relative" only, a per-timestep anchor of shape (seq_len, 9 * num_pose) is also
            accepted, which anchors each element of `pose` to the corresponding anchor.
        pose_repr (str): One of `POSE_REPR_LIST`.
        backward (bool): If True, convert from the given representation back to the absolute one.

    Returns:
        np.ndarray: Converted pose with the same shape as `pose`.
    """
    if pose_repr not in POSE_REPR_LIST:
        raise ValueError(
            f"[convert_pose_repr] Invalid pose representation '{pose_repr}'. Expected one of {POSE_REPR_LIST}."
        )

    pose = np.asarray(pose)

    if pose_repr == "abs":
        return pose

    if pose.ndim not in (1, 2):
        raise ValueError(
            f"[convert_pose_repr] Pose must be 1 or 2 dimensional: {pose.shape}"
        )
    if pose.shape[-1] % 9 != 0:
        raise ValueError(
            f"[convert_pose_repr] Last dimension must be divisible by 9: {pose.shape}"
        )

    is_single = pose.ndim == 1
    pose_seq = pose[np.newaxis] if is_single else pose
    seq_len = pose_seq.shape[0]
    num_pose = pose_seq.shape[-1] // 9

    anchor_pose = np.asarray(anchor_pose)
    is_base_seq = anchor_pose.ndim == 2
    if is_base_seq and pose_repr != "relative":
        raise ValueError(
            f"[convert_pose_repr] Per-timestep anchor pose is only supported for 'relative': {pose_repr}"
        )
    expected_base_shape = (seq_len, 9 * num_pose) if is_base_seq else (9 * num_pose,)
    if anchor_pose.shape != expected_base_shape:
        raise ValueError(
            f"[convert_pose_repr] Anchor pose shape must be {expected_base_shape}: {anchor_pose.shape}"
        )

    mat_seq = _get_mat_from_pose9(pose_seq.reshape(seq_len, num_pose, 9))
    anchor_mat = _get_mat_from_pose9(anchor_pose.reshape(-1, num_pose, 9))
    if not is_base_seq:
        anchor_mat = anchor_mat[0]

    if pose_repr == "relative":
        if backward:
            converted_mat_seq = anchor_mat @ mat_seq
        else:
            converted_mat_seq = np.linalg.inv(anchor_mat) @ mat_seq
    else:  # "delta", with translation and rotation handled separately (see docstring)
        converted_mat_seq = np.copy(mat_seq)
        pos_seq, rot_seq = mat_seq[..., :3, 3], mat_seq[..., :3, :3]
        anchor_pos, anchor_rot = anchor_mat[..., :3, 3], anchor_mat[..., :3, :3]

        if backward:
            converted_mat_seq[..., :3, 3] = np.cumsum(pos_seq, axis=0) + anchor_pos

            current_rot = anchor_rot
            for time_idx in range(seq_len):
                current_rot = rot_seq[time_idx] @ current_rot
                converted_mat_seq[time_idx, :, :3, :3] = current_rot
        else:
            all_pos = np.concatenate([anchor_pos[np.newaxis], pos_seq], axis=0)
            converted_mat_seq[..., :3, 3] = np.diff(all_pos, axis=0)

            all_rot = np.concatenate([anchor_rot[np.newaxis], rot_seq], axis=0)
            converted_mat_seq[..., :3, :3] = all_rot[1:] @ np.linalg.inv(all_rot[:-1])

    converted_pose = _get_pose9_from_mat(converted_mat_seq).reshape(
        seq_len, 9 * num_pose
    )

    return converted_pose[0] if is_single else converted_pose
