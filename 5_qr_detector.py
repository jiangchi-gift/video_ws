import logging
from typing import List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from pyzbar import pyzbar  # type: ignore
    _HAS_PYZBAR = True
except Exception:
    pyzbar = None
    _HAS_PYZBAR = False


class QRDetector:
    """二维码检测器：优先使用 pyzbar，失败时回退到 OpenCV 的 QRCodeDetector。

    detect() 返回列表，每项包含 'content'(str), 'points'(ndarray 4x2), 'method'(str)
    """

    def __init__(self, use_pyzbar: bool = True):
        self.use_pyzbar = use_pyzbar and _HAS_PYZBAR
        if use_pyzbar and not _HAS_PYZBAR:
            logger.warning("pyzbar 未找到，回退到 OpenCV 的 QRCodeDetector")
        if not self.use_pyzbar:
            self._cv_detector = cv2.QRCodeDetector()

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None:
            return []
        detected = []
        if self.use_pyzbar:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar.decode(gray)
            for i, barcode in enumerate(barcodes):
                if barcode.type == 'QRCODE':
                    try:
                        content = barcode.data.decode('utf-8')
                    except Exception:
                        content = barcode.data
                    x, y, w, h = barcode.rect
                    points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
                    detected.append({'content': content, 'points': points, 'method': 'pyzbar'})
            return detected

        # OpenCV fallback
        data, points, straight_qrcode = self._cv_detector.detectAndDecodeMulti(frame)
        if points is None:
            return []
        for idx, pts in enumerate(points):
            content = data[idx] if isinstance(data, (list, tuple)) and idx < len(data) else ''
            pts_arr = pts.reshape(-1, 2).astype(np.int32)
            detected.append({'content': content, 'points': pts_arr, 'method': 'opencv'})
        return detected


__all__ = ["QRDetector"]


def detect_first(frame: np.ndarray, use_pyzbar: bool = True):
    """快速检测并返回第一个结果的 content（或 None）。"""
    d = QRDetector(use_pyzbar=use_pyzbar)
    res = d.detect(frame)
    if not res:
        return None
    return res[0]


def decode_pyzbar(frame: np.ndarray):
    """如果 pyzbar 可用，返回 pyzbar.decode 的原始输出，否则抛出 ImportError。"""
    if not _HAS_PYZBAR:
        raise ImportError("pyzbar is not available")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return pyzbar.decode(gray)


__all__.extend(["detect_first", "decode_pyzbar"])
