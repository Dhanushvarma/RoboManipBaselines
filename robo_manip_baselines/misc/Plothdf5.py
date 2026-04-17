import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

dataset_dir_path = (
    "/home/dhanush/rmb_dev/RoboManipBaselines/robo_manip_baselines/dataset"
)
file_path = (
    dataset_dir_path
    + "/MujocoCallMTrash_20260417_130627_under_30/MujocoCallMTrash_world1_031.rmb/main.rmb.hdf5"
)

save_dir = "./plots"
os.makedirs(save_dir, exist_ok=True)


# ---- helper ----
def plot_1d(ax, x, title):
    ax.plot(x, linewidth=1)
    ax.set_title(title)
    ax.grid(True)


def plot_multi(ax, x, title):
    for i in range(x.shape[1]):
        ax.plot(x[:, i], label=f"dim {i}", linewidth=1)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


with h5py.File(file_path, "r") as f:

    # -------- pick signals you care about --------
    keys_1d = [
        "reward",
        "command_gripper_joint_pos",
        "measured_gripper_joint_pos",
    ]

    keys_nd = [
        "command_eef_pose",
        "measured_eef_pose",
        "command_joint_pos",
        "measured_joint_pos",
        "command_mobile_omni_vel",
        "measured_mobile_omni_vel",
        # rel
        "command_eef_pose_rel",
        "command_gripper_joint_pos_rel",
        "command_joint_pos_rel",
        "measured_eef_pose_rel",
        "measured_gripper_joint_pos_rel",
        "measured_joint_pos_rel",
    ]

    # -------- plot 1D signals --------
    for k in keys_1d:
        if k in f:
            data = f[k][:]
            fig, ax = plt.subplots(figsize=(8, 3))
            plot_1d(ax, data, k)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{k}.png"), dpi=150)
            plt.close()

    # -------- plot multi-dim signals --------
    for k in keys_nd:
        if k in f:
            data = f[k][:]
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_multi(ax, data, k)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{k}.png"), dpi=150)
            plt.close()

print(f"Saved plots to: {save_dir}")
