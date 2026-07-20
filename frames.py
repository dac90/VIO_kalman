"""Convert orientation quaternions between frames using a sensor.yaml's T_BS extrinsic.

EuRoC sensor.yaml files define T_BS as the transform from a sensor frame S
(camera, IMU, ...) into the body frame B: p_B = T_BS @ [p_S; 1]. Given an
orientation quaternion expressed in one of {body, sensor}, these functions
re-express it in the other, using the world-from-frame convention from
quaternion_utils (q rotates vectors from its frame into the world frame).
"""
import numpy as np
import yaml

from quaternion_utils import quat_conjugate, quat_multiply, rotate_vector_by_quaternion, rotation_matrix_to_quaternion


def load_T_BS(sensor_yaml_path):
    """Load the sensor-to-body extrinsic transform T_BS from an EuRoC sensor.yaml file.

    Returns the 4x4 homogeneous transform such that p_B = T_BS @ [p_S; 1]:
    T_BS rotates/translates a point from the sensor frame S into the body frame B.
    """
    with open(sensor_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    t_bs = cfg["T_BS"]
    return np.array(t_bs["data"], dtype=float).reshape(t_bs["rows"], t_bs["cols"])


def body_to_sensor_quaternion(q_world_body, T_BS):
    """Re-express a world-from-body orientation quaternion as world-from-sensor.

    T_BS's rotation block rotates vectors from the sensor frame into the body
    frame, so composing it with q_world_body (body's rotation into world)
    gives the sensor's orientation in world coordinates.
    """
    q_body_sensor = rotation_matrix_to_quaternion(T_BS[:3, :3])
    return quat_multiply(q_world_body, q_body_sensor)


def sensor_to_body_quaternion(q_world_sensor, T_BS):
    """Inverse of body_to_sensor_quaternion: world-from-sensor -> world-from-body."""
    q_body_sensor = rotation_matrix_to_quaternion(T_BS[:3, :3])
    return quat_multiply(q_world_sensor, quat_conjugate(q_body_sensor))


def body_to_sensor_position(p_world_body, q_world_body, T_BS):
    """World-frame position of the sensor origin, given the body's world position/orientation.

    p_S_world = p_B_world + R(q_world_body) @ t_BS, where t_BS is the sensor
    origin's position expressed in body-frame coordinates (T_BS's translation
    column).
    """
    t_bs = T_BS[:3, 3]
    return p_world_body + rotate_vector_by_quaternion(q_world_body, t_bs)
