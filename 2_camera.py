import logging
import threading
import time
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class Camera:
    """更健壮的摄像头封装：支持重连、超时读取、上下文管理和属性查询。

    注意：在多线程环境下尽量通过单一 Camera 实例访问设备。
    """

    def __init__(self, camera_id: int = 0, backend: int = cv2.CAP_DSHOW, reconnect_attempts: int = 3, reconnect_delay: float = 1.0):
        self.camera_id = camera_id
        self.backend = backend
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self._lock = threading.RLock()
        self.cap: Optional[cv2.VideoCapture] = None
        self.open()

    def open(self) -> None:
        with self._lock:
            attempts = 0
            while attempts < self.reconnect_attempts:
                self.cap = cv2.VideoCapture(self.camera_id, self.backend)
                if self.cap.isOpened():
                    logger.debug("摄像头打开成功 id=%s", self.camera_id)
                    return
                attempts += 1
                logger.warning("打开摄像头失败，重试 %s/%s", attempts, self.reconnect_attempts)
                time.sleep(self.reconnect_delay)
            raise RuntimeError(f"无法打开摄像头 id={self.camera_id} after {self.reconnect_attempts} attempts")

    def read(self, timeout: float = 2.0):
        """读取一帧；在超时或读取失败时返回 None。"""
        with self._lock:
            if self.cap is None or not self.cap.isOpened():
                try:
                    self.open()
                except Exception:
                    return None
            start = time.time()
            while True:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    timestamp = time.time()
                    return frame, timestamp
                if time.time() - start > timeout:
                    logger.warning("读取摄像头超时")
                    return None

    def read_frame(self, timeout: float = 2.0):
        """向后兼容方法：仅返回帧（旧代码使用 read_frame）。

        返回：frame 或 None
        """
        res = self.read(timeout=timeout)
        if res is None:
            return None
        # res 可能是 (frame, timestamp) 或 frame
        if isinstance(res, tuple):
            return res[0]
        return res

    def release(self) -> None:
        with self._lock:
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
            except Exception:
                logger.exception("释放摄像头失败")

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def set_resolution(self, width: int, height: int) -> None:
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def find_camera(max_index: int = 5) -> Tuple[Optional[int], Optional[int]]:
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        for i in range(max_index):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return i, backend
    return None, None


__all__ = ["Camera", "find_camera"]


# 兼容性辅助：返回已打开的 Camera 实例（旧代码可能使用 open_camera）
def open_camera(camera_id: int = 0, backend: int = cv2.CAP_DSHOW, reconnect_attempts: int = 3, reconnect_delay: float = 1.0) -> Camera:
    return Camera(camera_id=camera_id, backend=backend, reconnect_attempts=reconnect_attempts, reconnect_delay=reconnect_delay)


# 兼容别名：保留旧方法名
setattr(Camera, 'open_device', Camera.open)
setattr(Camera, 'readFrame', Camera.read_frame)
__all__.extend(['open_camera'])
