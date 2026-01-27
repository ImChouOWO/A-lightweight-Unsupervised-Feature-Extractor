import numpy as np
from filterpy.kalman import KalmanFilter


def bbox_xyxy_to_z(bbox):
    """
    bbox: (x1, y1, x2, y2)
    return z: (cx, cy, a, h)  where a = w / h
    """
    x1, y1, x2, y2 = map(float, bbox)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    a = w / h
    return np.array([cx, cy, a, h], dtype=np.float32)


def x_to_bbox_xyxy(x):
    """
    x: state vector (8,) = [cx, cy, a, h, vcx, vcy, va, vh]
    return bbox: (x1, y1, x2, y2)
    """
    cx, cy, a, h = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    h = max(h, 1.0)
    a = max(a, 1e-3)
    w = max(a * h, 1.0)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return (x1, y1, x2, y2)


def init_kf_from_bbox(
    bbox_xyxy,
    dt=1.0,
    std_pos=1.0,
    std_vel=10.0,
    std_meas_pos=1.0,
    std_meas_scale=1.0,
):
    """
    DeepSORT-style linear KF (8D state, 4D measurement):

      state x = [cx, cy, a, h, vcx, vcy, va, vh]^T
      meas  z = [cx, cy, a, h]^T

    Parameters (starting point; tune per FPS/resolution):
      dt            : frame time step
      std_pos       : process noise std for position-like states
      std_vel       : process noise std for velocity-like states
      std_meas_pos  : measurement noise std for (cx, cy)
      std_meas_scale: measurement noise std for (a, h)
    """
    kf = KalmanFilter(dim_x=8, dim_z=4)

    # State transition (constant velocity model)
    F = np.eye(8, dtype=np.float32)
    for i in range(4):
        F[i, i + 4] = dt
    kf.F = F

    # Measurement function (observe first 4 components)
    H = np.zeros((4, 8), dtype=np.float32)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    H[3, 3] = 1.0
    kf.H = H

    # Initialize state from bbox
    z0 = bbox_xyxy_to_z(bbox_xyxy)  # (cx, cy, a, h)
    kf.x = np.zeros((8, 1), dtype=np.float32)
    kf.x[0:4, 0] = z0

    # Initial covariance: high uncertainty for velocity terms
    P = np.eye(8, dtype=np.float32)
    P[0, 0] = 10.0
    P[1, 1] = 10.0
    P[2, 2] = 10.0
    P[3, 3] = 10.0
    P[4, 4] = 1000.0
    P[5, 5] = 1000.0
    P[6, 6] = 1000.0
    P[7, 7] = 1000.0
    kf.P = P

    # Process noise (Q): simple diagonal start (tune later)
    q = np.array(
        [std_pos, std_pos, std_pos, std_pos, std_vel, std_vel, std_vel, std_vel],
        dtype=np.float32,
    )
    kf.Q = np.diag(q * q)

    # Measurement noise (R)
    r = np.array([std_meas_pos, std_meas_pos, std_meas_scale, std_meas_scale], dtype=np.float32)
    kf.R = np.diag(r * r)

    return kf


# Optional: gating distance (Mahalanobis) helper for association
def gating_distance_maha(kf: KalmanFilter, bbox_xyxy):
    """
    Return squared Mahalanobis distance d^2 between measurement z and predicted measurement.
    Smaller means more likely match.
    """
    z = bbox_xyxy_to_z(bbox_xyxy).reshape(4, 1).astype(np.float32)
    y = z - (kf.H @ kf.x)                      # innovation
    S = kf.H @ kf.P @ kf.H.T + kf.R            # innovation covariance
    # robust inverse
    Sinv = np.linalg.inv(S + 1e-9 * np.eye(4, dtype=np.float32))
    d2 = float((y.T @ Sinv @ y)[0, 0])
    return d2
