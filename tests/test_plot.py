import numpy as np

from plot import load_imu_data


def test_load_imu_data_shapes_and_no_nans():
    gyro, accel = load_imu_data()
    assert gyro.shape[1] == 3
    assert accel.shape[1] == 3
    assert len(gyro) == len(accel)
    assert not np.isnan(gyro).any()
    assert not np.isnan(accel).any()


def test_load_imu_data_mean_matches_known_gravity_direction():
    # regression check against the gravity-direction analysis from earlier in
    # the project: accel mean should be dominated by its X component (~9.1)
    _, accel = load_imu_data()
    mean_accel = accel.mean(axis=0)
    assert mean_accel[0] > 8.0
    assert abs(mean_accel[1]) < 1.0
