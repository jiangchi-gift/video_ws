import logging
from typing import Tuple, Union, Iterable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def create_default_camera_params():
    """返回示例相机参数（请用实际标定值替换）。

    返回：camera_matrix, dist_coeffs, R, t, depth
    """
    camera_matrix = np.array([
        [1000, 0, 960],
        [0, 1000, 540],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array([0.1, -0.05, 0.001, 0.001, 0.0], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.zeros((3, 1), dtype=np.float64)
    depth = 0.5
    return camera_matrix, dist_coeffs, R, t, depth


def pixel_to_world(
    pixel_x: Union[float, Iterable[float]],
    pixel_y: Union[float, Iterable[float]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    depth: float
) -> Union[Tuple[float, float, float], np.ndarray]:
    """将像素坐标反投影到世界坐标。

    支持单点或多点（向量化）。当 depth 为 None 时假设目标在 Z=0 平面并尝试用射线与平面求交（需要已知 R,t）。
    """
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    single = np.isscalar(pixel_x) and np.isscalar(pixel_y)

    pts = None
    if single:
        pts = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float64)
    else:
        px = np.asarray(pixel_x, dtype=np.float64).ravel()
        py = np.asarray(pixel_y, dtype=np.float64).ravel()
        pts = np.stack([px, py], axis=1).reshape(-1, 1, 2)

    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    # undistorted shape: (N,1,2)
    undistorted = undistorted.reshape(-1, 2)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    Xc = (undistorted[:, 0] - cx) * depth / fx
    Yc = (undistorted[:, 1] - cy) * depth / fy
    Zc = np.full_like(Xc, fill_value=depth)

    camera_points = np.vstack([Xc, Yc, Zc]).T.reshape(-1, 3, 1)
    world = (R @ camera_points) + t.reshape(1, 3, 1)
    world = world.reshape(-1, 3)

    if single:
        Xw, Yw, Zw = world[0]
        return float(Xw), float(Yw), float(Zw)
    else:
        return world


__all__ = ["create_default_camera_params", "pixel_to_world"]


def pixel_to_world_xy(x, y, camera_matrix, dist_coeffs, R, t, depth):
    """兼容别名：接受单点 x,y 或数组化输入，转发到 `pixel_to_world`。"""
    return pixel_to_world(x, y, camera_matrix, dist_coeffs, R, t, depth)


def safe_create_default_camera_params():
    """别名：与 `create_default_camera_params` 等价，用于旧代码兼容。"""
    return create_default_camera_params()


__all__.extend(["pixel_to_world_xy", "safe_create_default_camera_params"])
