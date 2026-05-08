import cv2
import numpy as np
from typing import Tuple


def create_default_camera_params():
    """创建默认的相机内外参（示例参数，请按实际标定值替换）"""
    # 内参矩阵 (camera_matrix) - 假设分辨率 1920x1080，焦距约 1000 像素
    camera_matrix = np.array([
        [1000, 0, 960],
        [0, 1000, 540],
        [0, 0, 1]
    ], dtype=np.float64)

    # 畸变系数
    dist_coeffs = np.array([0.1, -0.05, 0.001, 0.001, 0.0], dtype=np.float64)

    # 外参：世界坐标系 Z=0 平面，相机在 (0, 0, 0.5) 米，光轴垂直向下
    R = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float64)

    t = np.array([0, 0, 0.5], dtype=np.float64).reshape(3, 1)

    depth = 0.5  # 相机光心到目标平面的垂直距离（米）

    return camera_matrix, dist_coeffs, R, t, depth


def pixel_to_world(
    pixel_x: float,
    pixel_y: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    depth: float
) -> Tuple[float, float, float]:
    """
    将像素坐标 (pixel_x, pixel_y) 转换为世界坐标 (Xw, Yw, Zw=0)
    :param pixel_x: 像素 x
    :param pixel_y: 像素 y
    :param camera_matrix: 相机内参矩阵 (3x3)
    :param dist_coeffs: 畸变系数 (1x5)
    :param R: 旋转矩阵 (3x3)
    :param t: 平移向量 (3x1)
    :param depth: 相机坐标系下目标的 Zc 深度（米）
    :return: (Xw, Yw, Zw) 世界坐标，单位米
    """
    # 1. 畸变校正
    undistorted_points = cv2.undistortPoints(
        np.array([[[pixel_x, pixel_y]]], dtype=np.float64),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix
    )
    u_undistorted, v_undistorted = undistorted_points[0][0]

    # 2. 反投影到相机坐标系
    # 已知 Zc = depth (固定平面深度)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    Xc = (u_undistorted - cx) * depth / fx
    Yc = (v_undistorted - cy) * depth / fy
    Zc = depth

    camera_point = np.array([Xc, Yc, Zc], dtype=np.float64).reshape(3, 1)

    # 3. 转换到世界坐标系
    world_point = R @ camera_point + t
    Xw, Yw, Zw = world_point.flatten()

    return Xw, Yw, Zw