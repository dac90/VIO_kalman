"""Time-coloured 3D scatter plots of the EuRoC imu0 data (gyro + accel)."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAV0_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "machine_hall", "MH_01_easy", "MH_01_easy", "mav0")

GYRO_COLS = ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]
ACCEL_COLS = ["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]


def load_imu_data(mav0_dir=MAV0_DIR):
    path = os.path.join(mav0_dir, "imu0", "data.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    gyro = df[GYRO_COLS].to_numpy()
    accel = df[ACCEL_COLS].to_numpy()
    return gyro, accel


def plot_time_coloured_3d(ax, data, axis_labels, title):
    """Scatter data (N,3) in 3D, coloured by sample index, with its own colourbar."""
    indices = np.arange(len(data))
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=indices, cmap="viridis", s=5)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)
    ax.figure.colorbar(scatter, ax=ax, label="Sample index", shrink=0.6, pad=0.1)
    return scatter


if __name__ == "__main__":
    gyro, accel = load_imu_data()

    print("Mean gyro  [x, y, z] (rad/s):", gyro.mean(axis=0))
    print("Mean accel [x, y, z] (m/s^2):", accel.mean(axis=0))

    fig = plt.figure(figsize=(14, 7))
    ax_gyro = fig.add_subplot(121, projection="3d")
    ax_accel = fig.add_subplot(122, projection="3d")

    plot_time_coloured_3d(ax_gyro, gyro, ["Gyro X (rad/s)", "Gyro Y (rad/s)", "Gyro Z (rad/s)"],
                          "Gyroscope 3D Scatter Plot")
    plot_time_coloured_3d(ax_accel, accel, ["Accel X (m/s^2)", "Accel Y (m/s^2)", "Accel Z (m/s^2)"],
                          "Accelerometer 3D Scatter Plot")

    fig.suptitle("MH_01_easy imu0: Angular Velocity and Linear Acceleration")
    plt.show()
