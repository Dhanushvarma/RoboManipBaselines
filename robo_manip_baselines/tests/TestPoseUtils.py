import numpy as np
from scipy.spatial.transform import Rotation

from robo_manip_baselines.common import convert_pose_repr
from robo_manip_baselines.common.utils.PoseUtils import (
    _get_mat_from_pose9,
    _get_pose9_from_mat,
)

TOLERANCE = 1e-9


def get_random_pose9(num_step, num_eef=1, seed=0):
    rng = np.random.default_rng(seed)
    rot = Rotation.random(num_step * num_eef, random_state=seed).as_matrix()
    mat = np.zeros((num_step, num_eef, 4, 4))
    mat[..., :3, :3] = rot.reshape(num_step, num_eef, 3, 3)
    mat[..., :3, 3] = rng.normal(size=(num_step, num_eef, 3))
    mat[..., 3, 3] = 1.0

    return _get_pose9_from_mat(mat).reshape(num_step, 9 * num_eef)


def test_round_trip(num_step=12):
    """The backward pass must recover the absolute pose for every representation."""
    for num_eef in (1, 2):
        pose_seq = get_random_pose9(num_step, num_eef)
        anchor_pose = get_random_pose9(1, num_eef, seed=1)[0]

        for pose_repr in ("abs", "relative", "delta"):
            converted = convert_pose_repr(pose_seq, anchor_pose, pose_repr)
            restored = convert_pose_repr(
                converted, anchor_pose, pose_repr, backward=True
            )
            error = np.abs(restored - pose_seq).max()
            assert error < TOLERANCE, f"{pose_repr} (num_eef={num_eef}): {error}"
            print(f"[test_round_trip] {pose_repr} (num_eef={num_eef}) error: {error}")


def test_relative_is_anchored_to_base(num_step=12):
    """"relative" must express every step in the single shared anchor frame."""
    pose_seq = get_random_pose9(num_step)
    anchor_pose = get_random_pose9(1, seed=1)[0]

    converted = convert_pose_repr(pose_seq, anchor_pose, "relative")

    expected = np.linalg.inv(_get_mat_from_pose9(anchor_pose)) @ _get_mat_from_pose9(
        pose_seq
    )
    error = np.abs(_get_pose9_from_mat(expected) - converted).max()
    assert error < TOLERANCE, error
    print(f"[test_relative_is_anchored_to_base] error: {error}")


def test_delta_is_anchored_to_previous_step(num_step=12):
    """
    "delta" must use UMI's world-frame convention, not a rigid SE(3) composition.

    Translation is a plain world-frame difference while rotation is left-multiplied, so the two are
    decoupled. Anchoring to the previous step rather than a shared base is also what separates "delta"
    from "relative", so the two must not coincide.
    """
    pose_seq = get_random_pose9(num_step)
    anchor_pose = get_random_pose9(1, seed=1)[0]

    converted = convert_pose_repr(pose_seq, anchor_pose, "delta")

    mat_seq = _get_mat_from_pose9(pose_seq)
    anchor_mat = _get_mat_from_pose9(anchor_pose)
    expected = np.copy(mat_seq)
    all_pos = np.concatenate([anchor_mat[np.newaxis, :3, 3], mat_seq[:, :3, 3]], axis=0)
    expected[:, :3, 3] = np.diff(all_pos, axis=0)
    all_rot = np.concatenate(
        [anchor_mat[np.newaxis, :3, :3], mat_seq[:, :3, :3]], axis=0
    )
    expected[:, :3, :3] = all_rot[1:] @ np.linalg.inv(all_rot[:-1])

    error = np.abs(_get_pose9_from_mat(expected) - converted).max()
    assert error < TOLERANCE, error
    print(f"[test_delta_is_anchored_to_previous_step] error: {error}")

    difference = np.abs(
        converted - convert_pose_repr(pose_seq, anchor_pose, "relative")
    ).max()
    assert difference > TOLERANCE, difference
    print(f"[test_delta_is_anchored_to_previous_step] relative vs delta: {difference}")


def test_single_step_matches_sequence(num_step=12):
    """A single pose must convert identically to the same pose inside a sequence."""
    pose_seq = get_random_pose9(num_step)
    anchor_pose = get_random_pose9(1, seed=1)[0]

    converted = convert_pose_repr(pose_seq, anchor_pose, "relative")
    single = convert_pose_repr(pose_seq[0], anchor_pose, "relative")

    assert single.shape == (9,), single.shape
    error = np.abs(single - converted[0]).max()
    assert error < TOLERANCE, error
    print(f"[test_single_step_matches_sequence] error: {error}")


def test_per_step_anchor_pose(num_step=12):
    """A per-timestep base must anchor each step to its own base."""
    pose_seq = get_random_pose9(num_step)
    anchor_pose_seq = get_random_pose9(num_step, seed=1)

    converted = convert_pose_repr(pose_seq, anchor_pose_seq, "relative")

    expected = np.stack(
        [
            convert_pose_repr(pose, anchor_pose, "relative")
            for pose, anchor_pose in zip(pose_seq, anchor_pose_seq)
        ]
    )
    error = np.abs(expected - converted).max()
    assert error < TOLERANCE, error
    print(f"[test_per_step_anchor_pose] error: {error}")


if __name__ == "__main__":
    test_round_trip()
    test_relative_is_anchored_to_base()
    test_delta_is_anchored_to_previous_step()
    test_single_step_matches_sequence()
    test_per_step_anchor_pose()
