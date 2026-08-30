import argparse

import numpy as np

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_pose_repr,
    get_pose9_from_pose7,
    get_pose_from_se3,
    get_rel_pose_from_se3,
    get_se3_from_pose,
    get_se3_from_rel_pose,
    get_skipped_data_seq,
)


def add_arrays_to_smaller_length(arr1, arr2):
    min_length = min(arr1.shape[0], arr2.shape[0])
    return arr1[:min_length] + arr2[:min_length]


def test_get_skipped_data_seq_joint_pos(filename, skip=3):
    abs_key = DataKey.MEASURED_JOINT_POS
    rel_key = DataKey.get_rel_key(abs_key)

    with RmbData(filename) as rmb_data:
        joint_abs_seq = rmb_data[abs_key][:]
        joint_rel_seq = rmb_data[rel_key][:]

    skipped_joint_abs_seq = joint_abs_seq[::skip][1:]

    skipped_joint_rel_seq = get_skipped_data_seq(joint_rel_seq[1:], rel_key, skip)

    skipped_joint_abs_seq2 = add_arrays_to_smaller_length(
        joint_abs_seq[::skip], skipped_joint_rel_seq
    )

    error = np.sum(
        np.abs(
            add_arrays_to_smaller_length(
                skipped_joint_abs_seq, -1 * skipped_joint_abs_seq2
            )
        )
    )

    print(f"[test_get_skipped_data_seq_joint_pos] error: {error}")


def test_get_skipped_data_seq_eef_pose(filename, skip=3):
    abs_key = DataKey.MEASURED_EEF_POSE
    rel_key = DataKey.get_rel_key(abs_key)

    with RmbData(filename) as rmb_data:
        eef_abs_seq = rmb_data[abs_key][:]
        eef_rel_seq = rmb_data[rel_key][:]

    skipped_eef_abs_seq = eef_abs_seq[::skip][1:]

    skipped_eef_rel_seq = get_skipped_data_seq(eef_rel_seq[1:], rel_key, skip)

    skipped_eef_abs_seq2 = []
    for eef_abs, eef_rel in zip(eef_abs_seq[::skip][:-1], skipped_eef_rel_seq):
        skipped_eef_abs_seq2.append(
            get_pose_from_se3(
                get_se3_from_pose(eef_abs) * get_se3_from_rel_pose(eef_rel)
            )
        )
    skipped_eef_abs_seq2 = np.array(skipped_eef_abs_seq2)

    error = np.sum(
        np.abs(
            add_arrays_to_smaller_length(skipped_eef_abs_seq, -1 * skipped_eef_abs_seq2)
        )
    )

    print(f"[test_get_skipped_data_seq_eef_pose] error: {error}")


def test_eef_pose_rel_key_is_per_step_delta(filename):
    """
    `MEASURED_EEF_POSE_REL` must be a per-step body-frame delta, not a base-anchored relative pose.

    This is the distinction the `_rel` suffix hides: the key is `inv(T_{t-1}) @ T_t` (see
    `DataManager.calc_rel_data`), so it is a *delta* in UMI terminology. `convert_pose_repr`'s
    "relative" is the base-anchored representation, and the two must not coincide.
    """
    abs_key = DataKey.MEASURED_EEF_POSE
    rel_key = DataKey.get_rel_key(abs_key)

    with RmbData(filename) as rmb_data:
        eef_abs_seq = rmb_data[abs_key][:]
        eef_rel_seq = rmb_data[rel_key][:]

    num_eef = eef_abs_seq.shape[-1] // 7
    expected_rel_seq = np.array(
        [
            np.concatenate(
                [
                    get_rel_pose_from_se3(
                        get_se3_from_pose(prev).actInv(get_se3_from_pose(curr))
                    )
                    for prev, curr in zip(
                        prev_pose.reshape(num_eef, 7), curr_pose.reshape(num_eef, 7)
                    )
                ]
            )
            for prev_pose, curr_pose in zip(eef_abs_seq[:-1], eef_abs_seq[1:])
        ]
    )

    error = np.abs(expected_rel_seq - eef_rel_seq[1:]).max()
    print(f"[test_eef_pose_rel_key_is_per_step_delta] body-frame delta error: {error}")

    pose9_seq = get_pose9_from_pose7(eef_abs_seq[1:])
    anchor_pose = get_pose9_from_pose7(eef_abs_seq[0])
    difference = np.abs(
        convert_pose_repr(pose9_seq, anchor_pose, "relative")
        - convert_pose_repr(pose9_seq, anchor_pose, "delta")
    ).max()
    print(f"[test_eef_pose_rel_key_is_per_step_delta] relative vs delta: {difference}")


def test_convert_pose_repr_round_trip(filename):
    """The backward pass must recover the absolute pose sequence."""
    abs_key = DataKey.MEASURED_EEF_POSE

    with RmbData(filename) as rmb_data:
        eef_abs_seq = rmb_data[abs_key][:]

    pose9_seq = get_pose9_from_pose7(eef_abs_seq)
    anchor_pose = pose9_seq[0]

    for pose_repr in ("abs", "relative", "delta"):
        converted = convert_pose_repr(pose9_seq, anchor_pose, pose_repr)
        restored = convert_pose_repr(converted, anchor_pose, pose_repr, backward=True)
        error = np.abs(restored - pose9_seq).max()
        print(f"[test_convert_pose_repr_round_trip] {pose_repr} error: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        type=str,
        help="filename of teleoperation data",
    )
    args = parser.parse_args()

    test_get_skipped_data_seq_joint_pos(args.filename)
    test_get_skipped_data_seq_eef_pose(args.filename)
    test_eef_pose_rel_key_is_per_step_delta(args.filename)
    test_convert_pose_repr_round_trip(args.filename)
