import json
import logging
from typing import Tuple, Sequence, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def hsv_mask_from_image(img: np.ndarray, lower: Sequence[int], upper: Sequence[int], morph: bool = True, ksize: int = 5) -> np.ndarray:
    """生成 HSV 掩码；支持可选形态学降噪以应对嘈杂环境。"""
    if img is None:
        raise ValueError("img 为空")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(lower, dtype=np.int32)
    upper = np.array(upper, dtype=np.int32)
    mask = cv2.inRange(hsv, lower, upper)
    if morph:
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def calibrate_from_image(image_path: str, save_to: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片 {image_path}")

    window = "Calibration"
    cv2.namedWindow(window)
    cv2.createTrackbar("H min", window, 0, 179, lambda x: None)
    cv2.createTrackbar("H max", window, 179, 179, lambda x: None)
    cv2.createTrackbar("S min", window, 0, 255, lambda x: None)
    cv2.createTrackbar("S max", window, 255, 255, lambda x: None)
    cv2.createTrackbar("V min", window, 0, 255, lambda x: None)
    cv2.createTrackbar("V max", window, 255, 255, lambda x: None)

    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("H min", window)
        h_max = cv2.getTrackbarPos("H max", window)
        s_min = cv2.getTrackbarPos("S min", window)
        s_max = cv2.getTrackbarPos("S max", window)
        v_min = cv2.getTrackbarPos("V min", window)
        v_max = cv2.getTrackbarPos("V max", window)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = hsv_mask_from_image(img, lower, upper)

        cv2.imshow("Original", img)
        cv2.imshow("Mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and save_to:
            with open(save_to, 'w', encoding='utf-8') as f:
                json.dump({'lower': lower.tolist(), 'upper': upper.tolist()}, f, ensure_ascii=False, indent=2)
            logger.info("已保存阈值到 %s", save_to)

    cv2.destroyAllWindows()
    return lower, upper


def calibrate_from_camera(camera_id: int = 0, save_to: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    window = "Calibration"
    cv2.namedWindow(window)
    cv2.createTrackbar("H min", window, 0, 179, lambda x: None)
    cv2.createTrackbar("H max", window, 179, 179, lambda x: None)
    cv2.createTrackbar("S min", window, 0, 255, lambda x: None)
    cv2.createTrackbar("S max", window, 255, 255, lambda x: None)
    cv2.createTrackbar("V min", window, 0, 255, lambda x: None)
    cv2.createTrackbar("V max", window, 255, 255, lambda x: None)

    lower = upper = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h_min = cv2.getTrackbarPos("H min", window)
        h_max = cv2.getTrackbarPos("H max", window)
        s_min = cv2.getTrackbarPos("S min", window)
        s_max = cv2.getTrackbarPos("S max", window)
        v_min = cv2.getTrackbarPos("V min", window)
        v_max = cv2.getTrackbarPos("V max", window)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = hsv_mask_from_image(frame, lower, upper)

        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and save_to:
            with open(save_to, 'w', encoding='utf-8') as f:
                json.dump({'lower': lower.tolist(), 'upper': upper.tolist()}, f, ensure_ascii=False, indent=2)
            logger.info("已保存阈值到 %s", save_to)

    cap.release()
    cv2.destroyAllWindows()
    if lower is None or upper is None:
        raise RuntimeError("未获得阈值")
    return lower, upper


__all__ = ["hsv_mask_from_image", "calibrate_from_image", "calibrate_from_camera"]


def hsv_mask(img: np.ndarray, lower: Sequence[int], upper: Sequence[int], morph: bool = True, ksize: int = 5) -> np.ndarray:
    """兼容别名，代理到 `hsv_mask_from_image`。"""
    return hsv_mask_from_image(img, lower, upper, morph=morph, ksize=ksize)


def save_thresholds(path: str, lower: Sequence[int], upper: Sequence[int]):
    data = {'lower': list(map(int, lower)), 'upper': list(map(int, upper))}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_thresholds(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data.get('lower', [])), np.array(data.get('upper', []))


def calibrate_from_file(path: str):
    """从保存的阈值文件加载 lower/upper 并返回 numpy 数组。"""
    return load_thresholds(path)


__all__.extend(["hsv_mask", "save_thresholds", "load_thresholds", "calibrate_from_file"])
